import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import argparse
import torch.nn as nn
import wandb
import transformers
import random
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import sed_scores_eval

from models.atstframe.atst_student import StudentSED
from helpers.augment import frame_shift, time_mask, mixup, filter_augmentation, mixstyle, RandomResizeCrop
from data_util.audioset_classes import as_strong_train_classes, as_strong_eval_classes
from helpers.encode import ManyHotEncoder
from data_util.audioset_strong import get_training_dataset, get_eval_dataset
from data_util.audioset_strong import get_temporal_count_balanced_sample_weights, get_uniform_sample_weights, \
    get_weighted_sampler
from helpers.utils import worker_init_fn
from models.atstframe.ATSTF_wrapper import ATSTWrapper
from models.prediction_wrapper import PredictionsWrapper

# v0.1

class PLModule(pl.LightningModule):
    def __init__(self, config, encoder,teacher_model,student_model):
        super().__init__()
        self.config = config
        self.encoder = encoder
        self.teacher = teacher_model
        self.student = student_model

        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        # prepare ingredients for knowledge distillation
        assert 0 <= config.distillation_loss_weight <= 1, "Lambda for Knowledge Distillation must be between 0 and 1."
        self.strong_loss = nn.BCEWithLogitsLoss()
        self.distill_loss = nn.KLDivLoss(reduction='batchmean')

        self.freq_warp = RandomResizeCrop((1, 1.0), time_scale=(1.0, 1.0))

        self.val_durations_df = pd.read_csv(f"resources/eval_durations.csv",
                                            sep=",", header=None, names=["filename", "duration"])
        self.val_predictions_strong = {}
        self.val_ground_truth = {}
        self.val_duration = {}
        self.val_loss = []

    def forward(self, batch):
        x = batch["audio"]
        mel = self.student_model.mel_forward(x)
        y_strong, _ = self.model(mel)
        return y_strong

    def get_optimizer(
            self, lr, adamw=False, weight_decay=0.01, betas=(0.9, 0.999)
    ):
        # we split the parameters into two groups, one for the pretrained model and one for the downstream model
        # we also split each of them into <=1 dimensional and >=2 dimensional parameters, so we can only
        # apply weight decay to the >=2 dimensional parameters, thus excluding biases and batch norms, an idea from NanoGPT
        params_leq1D = []
        params_geq2D = []

        for name, param in self.student.named_parameters():
            if param.requires_grad:
                if param.ndimension() >= 2:
                    params_geq2D.append(param)
                else:
                    params_leq1D.append(param)

        param_groups = [
            {'params': params_leq1D, 'lr': lr},
            {'params': params_geq2D, 'lr': lr, 'weight_decay': weight_decay},
        ]

        if weight_decay > 0:
            assert adamw
        assert len(param_groups) > 0
        if adamw:
            print(f"\nUsing adamw weight_decay={weight_decay}!\n")
            return torch.optim.AdamW(param_groups, lr=lr, betas=betas)
        return torch.optim.Adam(param_groups, lr=lr, betas=betas)

    def get_lr_scheduler(
            self,
            optimizer,
            num_training_steps,
            schedule_mode="cos",
            gamma: float = 0.999996,
            num_warmup_steps=20000,
            lr_end=2e-7,
    ):
        if schedule_mode in {"exp"}:
            return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
        if schedule_mode in {"cosine", "cos"}:
            return transformers.get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )
        if schedule_mode in {"linear"}:
            print("Linear schedule!")
            return transformers.get_polynomial_decay_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                power=1.0,
                lr_end=lr_end,
            )
        raise RuntimeError(f"schedule_mode={schedule_mode} Unknown.")

    def configure_optimizers(self):
        """
        This is the way pytorch lightening requires optimizers and learning rate schedulers to be defined.
        The specified items are used automatically in the optimization loop (no need to call optimizer.step() yourself).
        :return: dict containing optimizer and learning rate scheduler
        """
        optimizer = self.get_optimizer(self.config.max_lr, adamw=self.config.adamw,
                                       weight_decay=self.config.weight_decay)

        num_training_steps = self.trainer.estimated_stepping_batches

        scheduler = self.get_lr_scheduler(optimizer, num_training_steps,
                                          schedule_mode=self.config.schedule_mode,
                                          lr_end=self.config.lr_end)
        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1
        }
        return [optimizer], [lr_scheduler_config]

    def training_step(self, train_batch, batch_idx):
        """
        :param train_batch: contains one batch from train dataloader
        :param batch_idx
        :return: a dict containing at least loss that is used to update model parameters, can also contain
                    other items that can be processed in 'training_epoch_end' to log other metrics than loss
        """

        x = train_batch["audio"]
        labels = train_batch['strong']
        if 'pseudo_strong' in train_batch:
            pseudo_labels = train_batch['pseudo_strong']
        else:
            # create dummy pseudo labels
            pseudo_labels = torch.zeros_like(labels)
            assert self.config.distillation_loss_weight == 0

        mel = self.student.mel_forward(x)

        # time rolling
        if self.config.frame_shift_range > 0:
            mel, labels, pseudo_labels = frame_shift(
                mel,
                labels,
                pseudo_labels=pseudo_labels,
                net_pooling=self.encoder.net_pooling,
                shift_range=self.config.frame_shift_range
            )

        # mix up
        if self.config.mixup_p > random.random():
            mel, labels, pseudo_labels = mixup(
                mel,
                targets=labels,
                pseudo_strong=pseudo_labels
            )

        # mix style
        if self.config.mixstyle_p > random.random():
            mel = mixstyle(
                mel
            )

        # time masking
        if self.config.max_time_mask_size > 0:
            mel, labels, pseudo_labels = time_mask(
                mel,
                labels,
                pseudo_labels=pseudo_labels,
                net_pooling=self.encoder.net_pooling,
                max_mask_ratio=self.config.max_time_mask_size
            )

        # frequency masking
        if self.config.filter_augment_p > random.random():
            mel, _ = filter_augmentation(
                mel
            )

        # frequency warping
        if self.config.freq_warp_p > random.random():
            mel = mel.squeeze(1)
            mel = self.freq_warp(mel)
            mel = mel.unsqueeze(1)

        # forward through network; use strong head
        y_hat_strong_teacher, _ = self.teacher(mel)
        y_hat_strong_student, _ = self.student(mel)

        last_n_layers_teacher = self.teacher.get_intermediate_layers(
            mel,
            self.teacher.fake_length.to(mel).repeat(len(mel)),
            1,
            scene=False
        )
        last_n_layers_student = self.student.get_intermediate_layers(
            mel,
            self.teacher.fake_length.to(mel).repeat(len(mel)),
            1,
            scene=False
        )



        strong_supervised_loss = self.strong_loss(y_hat_strong_student, labels)

        if self.config.distillation_loss_weight > 0:
            strong_distillation_loss = self.strong_loss(last_n_layers_student, last_n_layers_teacher)
        else:
            raise ValueError(f"distillation_loss_weight must be larger than 0, got {self.config.distillation_loss_weight}")

        loss = self.config.distillation_loss_weight * strong_distillation_loss \
               + (1 - self.config.distillation_loss_weight) * strong_supervised_loss

        # logging
        self.log('epoch', self.current_epoch)
        for i, param_group in enumerate(self.trainer.optimizers[0].param_groups):
            self.log(f'trainer/lr_optimizer_{i}', param_group['lr'])
        self.log("train/loss", loss.detach().cpu(), prog_bar=True)
        self.log("train/strong_supervised_loss", strong_supervised_loss.detach().cpu())
        self.log("train/strong_distillation_loss", strong_distillation_loss.detach().cpu())

        return loss

def train(config):
    # Train Models on temporally-strong portion of AudioSet.
    atst = ATSTWrapper()
    teacher_model = PredictionsWrapper(atst, checkpoint="ATST-F_strong_1")
    student_model = StudentSED(num_classes=15)

    # logging is done using wandb
    wandb_logger = WandbLogger(
        project="kd_from_ptsed",
        notes="Using conformer to do knowledge distillation from atst-model in ptsed",
        tags=["AudioSet Strong", "Sound Event Detection", "Knowledge Disitillation"],
        config=config,
        name=config.experiment_name
    )

    # encoder manages encoding and decoding of model predictions
    encoder = ManyHotEncoder(as_strong_train_classes)

    train_set = get_training_dataset(encoder, wavmix_p=config.wavmix_p,
                                     pseudo_labels_file=config.pseudo_labels_file)
    eval_set = get_eval_dataset(encoder)

    if config.use_balanced_sampler:
        sample_weights = get_temporal_count_balanced_sample_weights(train_set, save_folder="resources")
    else:
        sample_weights = get_uniform_sample_weights(train_set)

    train_sampler = get_weighted_sampler(sample_weights, epoch_len=config.epoch_len)

    # train dataloader
    train_dl = DataLoader(dataset=train_set,
                          sampler=train_sampler,
                          worker_init_fn=worker_init_fn,
                          num_workers=config.num_workers,
                          batch_size=config.batch_size,
                          shuffle=False)

    # eval dataloader
    eval_dl = DataLoader(dataset=eval_set,
                         worker_init_fn=worker_init_fn,
                         num_workers=config.num_workers,
                         batch_size=config.batch_size)

    # create pytorch lightening module
    pl_module = PLModule(config, encoder,teacher_model,student_model)

    # create the pytorch lightening trainer by specifying the number of epochs to train, the logger,
    # on which kind of device(s) to train and possible callbacks
    trainer = pl.Trainer(max_epochs=config.n_epochs,
                         logger=wandb_logger,
                         accelerator='auto',
                         devices=config.num_devices,
                         precision=config.precision,
                         num_sanity_val_steps=0,
                         check_val_every_n_epoch=config.check_val_every_n_epoch
                         )

    # start training and validation for the specified number of epochs
    trainer.fit(pl_module, train_dl, eval_dl)

    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration Parser. ')

    # general
    parser.add_argument('--experiment_name', type=str, default="AudioSet_Strong")
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--num_devices', type=int, default=1)
    parser.add_argument('--precision', type=int, default=16)
    parser.add_argument('--evaluate', action='store_true', default=False)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=5)

    # model
    parser.add_argument('--model_name', type=str,
                        choices=["ATST-F", "BEATs", "fpasst", "M2D", "ASIT"] + \
                                [f"frame_mn{width}" for width in ["06", "10"]],
                        default="ATST-F")  # used also for training
    # "scratch" = no pretraining
    # "ssl" = SSL pre-trained
    # "weak" = AudioSet Weak pre-trained
    # "strong" = AudioSet Strong pre-trained
    parser.add_argument('--pretrained', type=str, choices=["scratch", "ssl", "weak", "strong"],
                        default="weak")
    parser.add_argument('--seq_model_type', type=str, choices=["rnn"],
                        default=None)

    # training
    parser.add_argument('--n_epochs', type=int, default=30)
    parser.add_argument('--use_balanced_sampler', action='store_true', default=False)
    parser.add_argument('--distillation_loss_weight', type=float, default=0.0)
    parser.add_argument('--temperature', type=float, default=2.0)
    parser.add_argument('--epoch_len', type=int, default=100000)
    parser.add_argument('--median_window', type=int, default=9)

    # augmentation
    parser.add_argument('--wavmix_p', type=float, default=0.8)
    parser.add_argument('--freq_warp_p', type=float, default=0.8)
    parser.add_argument('--filter_augment_p', type=float, default=0.8)
    parser.add_argument('--frame_shift_range', type=float, default=0.125)  # in seconds
    parser.add_argument('--mixup_p', type=float, default=0.3)
    parser.add_argument('--mixstyle_p', type=float, default=0.3)
    parser.add_argument('--max_time_mask_size', type=float, default=0.0)

    # optimizer
    parser.add_argument('--adamw', action='store_true', default=False)
    parser.add_argument('--weight_decay', type=float, default=0.0)

    # lr schedule
    parser.add_argument('--schedule_mode', type=str, default="cos")
    parser.add_argument('--max_lr', type=float, default=7e-5)
    parser.add_argument('--lr_end', type=float, default=2e-7)
    parser.add_argument('--warmup_steps', type=int, default=5000)

    # knowledge distillation
    parser.add_argument('--pseudo_labels_file', type=str,
                        default=None)

    args = parser.parse_args()
    train(args)
