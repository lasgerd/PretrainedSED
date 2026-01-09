from resnet_conformer import *
from utils.utils import *
from tqdm import tqdm
import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader, random_split, DistributedSampler
import os
import time
from datetime import datetime
from datetime import timedelta
import sys
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from parameters import params

# v0.6.7

if __name__ == "__main__":

    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(1)
    import torch.multiprocessing as mp
    mp.set_start_method('spawn')  # 改为 'spawn' 而非默认 'fork'


    early_stopper = EarlyStopping(patience=params['patience_val_loss'])
    num_epochs = params['warmup_epochs'] + params['hold_epochs'] + params['decay_epochs']
    seld_net = ResNetConformerSELD(in_channels=params['in_channels'], num_classes=params['num_classes'],
                                   num_conformer_layers=params['num_conformer_layers'])

    # 总参数量
    total_params = sum(p.numel() for p in seld_net.parameters())
    print("SELD 总参数数量:", total_params)

    # 初始化进程组
    dist.init_process_group(backend="nccl",
                            timeout=timedelta(minutes=10))
    # 设置当前 GPU
    # local_rank = torch.distributed.get_rank()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(local_rank)
    print(f"My local_rank is {local_rank}")

    # 停止标志，放在 GPU 上
    stop_flag = torch.tensor([0], device=device)

    # 放到指定 GPU 上
    seld_net = seld_net.to(device)

    # 用 DDP 包装
    seld_net = DDP(seld_net, device_ids=[local_rank], output_device=local_rank)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(seld_net.parameters(), lr=params['learning_rate'])
    # 获取 scheduler
    scheduler = get_tri_stage_epoch_scheduler(optimizer, params['warmup_epochs'], params['hold_epochs'], params['decay_epochs'])

    # transform = torchaudio.transforms.MelSpectrogram(
    #     sample_rate=24000,
    #     n_fft=1024,
    #     win_length=960,
    #     hop_length=480,
    #     n_mels=64,
    #     center=True
    # )
    # transform = torchaudio.transforms.MFCC(
    #     sample_rate=24000,
    #     n_mfcc=64,
    #     log_mels=True
    # )

    transform = InterChannelTransform(
        sample_rate=params['sr'],
        n_fft=1024,
        win_length=960,
        hop_length=480,
        n_mels=64,
    )

    # transform = LogMelPhaseTransform(
    #     sample_rate=24000,
    #     n_fft=1024,
    #     win_length=960,
    #     hop_length=480,
    #     n_mels=64,
    # )

    # total_dataset = SELDDataset(audio_dir=params['current_main_folder'] + "/dataset/two_types_dataset/stereo",
    #                             label_dir=params['current_main_folder'] + "/dataset/two_types_dataset/metadata",
    #                             sample_rate=params['sr'],
    #                             num_frames = params['num_frames'],
    #                             transform=transform)

    total_dataset = LMDBSELDDataset("./dataset/lmdb/train_data.lmdb", sample_rate=params['sr'],
                            num_frames=params['num_frames'],
                            transform=transform)


    # Split dataset:
    total_len = len(total_dataset)
    train_len = int(total_len*params['train_ratio'])
    val_len = int(total_len*params['val_ratio'])
    test_len = int(total_len - val_len - train_len)

    train_dataset, val_dataset, test_dataset = random_split(total_dataset, [train_len,val_len, test_len],
                                             generator=torch.Generator().manual_seed(42))

    # 每个进程只加载自己分到的数据
    train_sampler = DistributedSampler(train_dataset, num_replicas=dist.get_world_size(), rank=rank, shuffle=True)

    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], sampler=train_sampler, shuffle=False, num_workers=params['num_cpu_workers'],
                              pin_memory=True, prefetch_factor=params['prefetch_factor'], drop_last=False, persistent_workers = True)

    # 验证集只在 rank0 上
    if rank == 0:
        test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False, num_workers=params['num_cpu_workers'], pin_memory=True,
                                 prefetch_factor=params['prefetch_factor'], persistent_workers = True)
        val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False, num_workers=params['num_cpu_workers'], pin_memory=True,
                                prefetch_factor=params['prefetch_factor'], persistent_workers = True)
    else:
        test_loader = None
        val_loader = None

    loss_list = []
    val_loss_list = []
    lr_list = []
    best_val_loss = float('inf')

    # Save LOG
    suffix = params['suffix']
    now = datetime.now()
    save_path = params['current_main_folder'] + "/outputs/" + now.strftime("%Y-%m-%d %H-%M") + suffix
    os.makedirs(save_path, exist_ok=True)
    best_model_path = save_path + "/best_model.pth"

    start_time = time.time()
    epoch_stop = num_epochs
    previous_loss = 1000
    patience_counter = 0

    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        seld_net.train()
        running_loss = 0.0
        current_lr = optimizer.param_groups[0]['lr']
        epoch_sed_loss = 0
        epoch_doa_loss = 0

        # 同步+广播
        dist.barrier()
        dist.broadcast(stop_flag, src=0)
        dist.barrier()

        if stop_flag.item() > 0:
            print(f"Rank {rank} cuda synchronizing")
            torch.cuda.synchronize()  # 等待所有 GPU 操作完成
            print(f"Rank {rank} exiting early")
            break  # 所有 rank 在此处安全同步退出

        loop = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}") if rank == 0 else train_loader

        # for batch in loop:
        #     inputs, label_accdoa,_,_ = batch
        #     inputs = inputs.to(device).permute(0, 1, 3, 2)
        #     label_accdoa = label_accdoa.to(device)

        #     optimizer.zero_grad()
        #     inputs, targets_a, targets_b, lam = mixup_data(inputs, label_accdoa, alpha=0.3)  # mixup
        #     accdoa_pred = seld_net(inputs)

        #     loss = lam * criterion(accdoa_pred, targets_a) + (1 - lam) * criterion(accdoa_pred, targets_b)

        #     loss.backward()
        #     optimizer.step()
        #     running_loss += loss.item() * inputs.size(0)

        for batch in loop:
            inputs, label_accdoa,_,_ = batch
            inputs = inputs.to(device).permute(0, 1, 3, 2) # [B,C,T,F] = [32,6,51,64]
            label_accdoa = label_accdoa.to(device)

            inputs = spec_augment(inputs, num_time_masks=1, time_mask_param=10, num_freq_masks=2, freq_mask_param=15)

            optimizer.zero_grad()
            inputs, targets_a, targets_b, lam = mixup_data(inputs, label_accdoa, alpha=0.4) #mixup
            accdoa_pred = seld_net(inputs)

            loss = lam * criterion(accdoa_pred, targets_a) + (1 - lam) * criterion(accdoa_pred, targets_b)
            # loss = criterion(accdoa_pred, label_accdoa)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        # epoch 结束，聚合
        total_loss_tensor = torch.tensor(running_loss, device=device)
        work = dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM, async_op=True)

        # 聚合样本数
        num_samples_tensor = torch.tensor(len(train_loader.sampler), device=device)
        dist.all_reduce(num_samples_tensor, op=dist.ReduceOp.SUM)

        epoch_loss = total_loss_tensor.item() / num_samples_tensor.item()
        scheduler.step()  # 每个 epoch 末尾调用一次

        if rank == 0:
            loss_list.append(epoch_loss)
            lr_list.append(current_lr)

            # Validation
            seld_net_eval = seld_net.module  # 此处取把模型从DDP里取出来，避免梯度的记录，很关键！！
            seld_net_eval = seld_net_eval.eval()
            val_running_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    inputs, label_accdoa,_,_ = batch
                    inputs = inputs.to(device).permute(0, 1, 3, 2)
                    label_accdoa = label_accdoa.to(device)

                    accdoa_pred = seld_net_eval(inputs)
                    loss_val = criterion(accdoa_pred, label_accdoa)

                    val_running_loss += loss_val.item() * inputs.size(0)

            val_loss = val_running_loss / len(val_loader.dataset)
            val_loss_list.append(val_loss)

            print(f"\nEpoch [{epoch + 1}/{num_epochs}], "
                  f"Train Loss: {epoch_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"current_lr: {current_lr:.6f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(seld_net_eval.state_dict(), best_model_path)
                print("✅ Best model updated.")

            # 提前退出条件1
            early_stopper(val_loss)
            if early_stopper.early_stop:
                print("Early stopping based on val loss triggered!")
                epoch_stop = epoch + 1
                stop_flag[0] = 1

            # 提前退出条件2
            if epoch_loss > previous_loss - 1e-4:
                patience_counter += 1
            else:
                patience_counter = 0

            if patience_counter >= params['patience_train_loss']:
                print("Training loss no longer improving. Stopping.")
                epoch_stop = epoch + 1
                stop_flag[0] = 1

            # 提前退出条件3:某个其他rank已经退出
            if stop_flag.item() == 1:
                print(f"Rank {rank} exiting because of stop_flag")
                stop_flag[0] = 1

    end_time = time.time()

    train_dur = end_time - start_time  # s

    if rank == 0:
        # Load best model for test
        seld_net_test = seld_net.module
        seld_net_test.load_state_dict(torch.load(best_model_path))
        seld_net_test.eval()

        # Test
        test_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                inputs, label_accdoa,_,_ = batch
                inputs = inputs.to(device).permute(0, 1, 3, 2)
                label_accdoa = label_accdoa.to(device)
                accdoa_pred = seld_net_test(inputs)
                loss_test = criterion(accdoa_pred, label_accdoa)

                test_loss += loss_test.item() * inputs.size(0)

        avg_test_loss = test_loss / len(test_loader.dataset)
        print(f"📊 Final Test Loss: {avg_test_loss:.4f}")

        # Save loss history
        now = datetime.now()
        save_info(save_path, train_dur, params['batch_size'], epoch_stop, params['warmup_epochs'],
                  params['hold_epochs'], params['decay_epochs'],
                  best_val_loss, params['patience_val_loss'], params['patience_train_loss'], avg_test_loss)
        np.savetxt(save_path + "/train_loss.txt", np.array(loss_list), fmt='%.6f')
        np.savetxt(save_path + "/val_loss.txt", np.array(val_loss_list), fmt='%.6f')
        np.savetxt(save_path + "/learning_rate.txt", np.array(lr_list), fmt='%.6f')

    print(f"📊 Rank {rank} synchronizing! ")
    torch.cuda.synchronize()  # 等待所有 GPU 操作完成
    print(f"📊 Rank {rank} finished! ")

    # 确保所有进程同步后再退出
    dist.barrier()

    print(f"📊 All rank synchronized! ")

    # 销毁进程组
    dist.destroy_process_group()