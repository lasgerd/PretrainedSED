import os
import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import LambdaLR
import os
import shutil
import numpy as np
import torch.nn.functional as F
import random
import pickle
import lmdb

# CLASS_LABELS = ['Accelerating_and_revving_and_vroom_and_racecar','Aircraft','Boom','carDrift','engine','explosion','glassBreak',
#                 'gun','Human_voice','Music','Smash','Vehicle']
# CLASS_LABELS = ['Accelerating_and_revving_and_vroom_and_racecar','Aircraft','Aircraft_engine','Car','carDrift','engine','Engine_knocking','Engine_starting','explosion','glassBreak',
#                 'gun','Human_voice','Music','Siren','Smash']
# CLASS_LABELS = ['Accelerating_and_revving_and_vroom_and_racecar','engine','explosion','Fire','glass',
#                 'gun','Human_voice','Knock','Music','Race_car','Siren','Smash']
# CLASS_LABELS = ['swing','blade2hand','boss_impact','user_impact','voice','others']
CLASS_LABELS = ['impact']
LABEL2IDX = {label: i for i, label in enumerate(CLASS_LABELS)}

class LMDBSELDDataset(Dataset):
    def __init__(self, lmdb_path, transform=None, num_frames=50, sample_rate=24000):
        self.lmdb_path = lmdb_path
        self.transform = transform
        self.sample_rate = sample_rate
        self.num_frames = num_frames
        self._env = None  # ⚠ 私有属性，不会被 pickle

    def _init_lmdb(self):
        if self._env is None:
            self._env = lmdb.open(
                self.lmdb_path,
                readonly=True,
                lock=False,
                readahead=True,
                meminit=False
            )

    def __len__(self):
        self._init_lmdb()
        with self._env.begin(write=False) as txn:
            return txn.stat()['entries']

    def __getitem__(self, idx):
        self._init_lmdb()
        with self._env.begin(write=False) as txn:
            key = f"{idx:06d}".encode('ascii')
            value = txn.get(key)
            waveform, label = pickle.loads(value)

        if self.transform:
            waveform = self.transform(waveform)

        num_frames = self.num_frames
        n_classes = len(LABEL2IDX)
        labels = torch.zeros((num_frames, n_classes), dtype=torch.float32)
        doa_target = torch.zeros((num_frames, n_classes), dtype=torch.float32)

        for i in range(label.shape[0]):
            frame_idx = int(label[i,0])
            class_idx = int(label[i,1])
            labels[frame_idx, class_idx] = 1.0
            doa_target[frame_idx, class_idx] = float(label[i,2])

        label_accdoa_x = labels * torch.cos(doa_target / 180 * torch.pi).abs()
        label_accdoa_y = labels * torch.sin(doa_target / 180 * torch.pi)
        label_accdoa = torch.stack([label_accdoa_x, label_accdoa_y], dim=2)

        return waveform, label_accdoa, doa_target, num_frames

    # ⚠ 关键：Windows spawn 时 pickle Dataset 不包含 env
    def __getstate__(self):
        state = self.__dict__.copy()
        state['_env'] = None
        return state

class SELDDataset(Dataset):
    def __init__(self, audio_dir, label_dir, num_frames = 50,sample_rate=24000, transform=None):
        self.audio_dir = audio_dir
        self.label_dir = label_dir
        self.sample_rate = sample_rate
        self.transform = transform
        self.num_frames = num_frames
        self.audio_files = sorted([f for f in os.listdir(audio_dir) if f.endswith('.wav')])

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        # 1. 加载音频
        audio_file = self.audio_files[idx]
        audio_path = os.path.join(self.audio_dir, audio_file)
        waveform, sr = torchaudio.load(audio_path)  # waveform: [C, T]
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        if self.transform:
            waveform = self.transform(waveform)

        # 2. 加载对应 CSV 标签
        label_path = os.path.join(self.label_dir, audio_file.replace('.wav', '.csv'))
        df = pd.read_csv(label_path)

        # 3. 构建帧级标签（举例：每帧有多个标签）
        # num_frames = waveform.shape[1] // 320  # 假设帧 hop=320（20ms at 16kHz）
        num_frames = self.num_frames #100ms一帧
        n_classes = len(LABEL2IDX)

        # 初始化 [frames, classes] 标签矩阵
        labels = torch.zeros((num_frames, n_classes), dtype=torch.float32)
        doa_target = torch.zeros((num_frames, n_classes), dtype=torch.float32)

        for _, row in df.iterrows():
            frame = int(row['frame'])
            if 0 <= frame < num_frames:
                # class_idx = LABEL2IDX[row['class']]
                class_idx = int(row['class'])
                labels[frame, class_idx] = 1.0

                doa_target[frame,class_idx] = row['sfx_azimuth'] # 0~360°，正前方为0°，逆时针旋转

        label_accdoa_x = labels * torch.cos(doa_target/180*torch.pi).abs() # 不区分前后，统一为正
        label_accdoa_y = labels * torch.sin(doa_target / 180 * torch.pi) # 区分左右
        label_accdoa = torch.stack([label_accdoa_x,label_accdoa_y],dim=2)

        return waveform, label_accdoa,doa_target,audio_file


class EarlyStopping:
    def __init__(self, patience=10, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        score = -val_loss  # 以最小化为目标
        if self.best_score is None or score > self.best_score + self.delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def save_info(save_path,train_dur,batch_size,epoch_stop,warmup_epochs, hold_epochs, decay_epochs,
              best_val_loss,patience_val_loss,patience_train_loss,avg_test_loss):
    text = [f"train_dur = {train_dur/60:.2f} min,\n",f"batch_size = {batch_size}\n",
            f"epoch_stop = {epoch_stop}\n",f"warmup_epochs = {warmup_epochs}\n",f"hold_epochs = {hold_epochs}\n",f"decay_epochs = {decay_epochs}\n",
            f"best_val_loss = {best_val_loss:.5f}\n",f"avg_test_loss = {avg_test_loss:.5f}\n",
            f"patience_val_loss = {patience_val_loss}\n",f"patience_train_loss = {patience_train_loss}\n"]
    with open(save_path+"/log.txt", "w", encoding="utf-8") as f:
        f.writelines(text)

def get_tri_stage_epoch_scheduler(optimizer, warmup_epochs, hold_epochs, decay_epochs):
    total_epochs = warmup_epochs + hold_epochs + decay_epochs

    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return (current_epoch + 1) / warmup_epochs
        elif current_epoch < warmup_epochs + hold_epochs:
            return 1.0
        elif current_epoch < total_epochs:
            decay_epoch = current_epoch - warmup_epochs - hold_epochs
            return max(0.0, (decay_epochs - decay_epoch) / decay_epochs)
        else:
            return 0.0

    return LambdaLR(optimizer, lr_lambda=lr_lambda)

def copy_folder_contents(src_dir, dst_dir):
    # 确保目标文件夹存在
    os.makedirs(dst_dir, exist_ok=True)

    for item in os.listdir(src_dir):
        s = os.path.join(src_dir, item)
        d = os.path.join(dst_dir, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)  # Python 3.8+
        else:
            shutil.copy2(s, d)  # 保留原文件修改时间等元信息


def mixup_data(x, y, alpha=0.3):
    '''返回混合后的输入和标签'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def spec_augment(feature,num_time_masks = 2,time_mask_param = 1,num_freq_masks = 2,freq_mask_param = 4):
    freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask_param)
    time_mask = torchaudio.transforms.TimeMasking(time_mask_param=time_mask_param)
    for _ in range(num_time_masks):
        feature = time_mask(feature)

    for _ in range(num_freq_masks):
        feature = freq_mask(feature)

    return feature


def doa_azimuth_loss(pred, target, mask=None):
    """
    pred: (B, T) 预测 azimuth (度)
    target: (B, T) GT azimuth，-90°~90°
    mask: (B, T) 活动帧掩码
    """
    # azimuth 转向量表示
    vec_pred = torch.stack([torch.cos(pred), torch.sin(pred)], dim=-1)  # (B, T, 2)
    vec_gt = torch.stack([torch.cos(target), torch.sin(target)], dim=-1)  # (B, T, 2)

    # 逐帧余弦相似度
    cos_sim = torch.sum(vec_pred * vec_gt, dim=-1)  # (B, T)

    # loss = (1 - cos_sim) / 2
    loss = (1 - cos_sim) * 0.5

    if mask is not None:
        loss = loss * mask
        loss = loss.sum() / (mask.sum() + 1e-6)
    else:
        loss = loss.mean()
    return loss


class LogMelTransform:
    def __init__(self, sample_rate=24000,n_fft=1024,win_length=960,hop_length=480,n_mels=64):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels

    def __call__(self, audio):
        transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            center=True
        )

        mel = transform(audio)
        log_mel = torch.log(mel)

        return log_mel


# 这部分计算方式，主要参考了：arXiv:2209.05900v1 [cs.SD] 13 Sep 2022，Binaural Signal Representations for Joint Sound
# Event Detection and Acoustic Scene Classification
class InterChannelTransform:
    def __init__(self, sample_rate=24000,n_fft=1024,win_length=960,hop_length=480,n_mels=64):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels

    def __call__(self, audio):
        transform_spectrum = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            power = None
        )
        mel_fb = torchaudio.functional.melscale_fbanks(
            int(self.n_fft // 2 + 1),
            n_mels=self.n_mels,
            f_min=0.0,
            f_max=self.sample_rate / 2.0,
            sample_rate=self.sample_rate,
        )
        spec = transform_spectrum(audio)
        spec_l,spec_r = spec[0,:,:],spec[1,:,:]

        # 1) Mel 幅度
        spec_mel_l = (spec_l.t() @ mel_fb.to(torch.complex64)).t()# [m,t]
        spec_mel_r = (spec_r.t() @ mel_fb.to(torch.complex64)).t() # [m,t]
        spec_mel_l_mag = spec_mel_l.abs()
        spec_mel_r_mag = spec_mel_r.abs()

        # 2) Mel ILD
        ILD = 20.0 * torch.log10((spec_mel_l.abs() + 1e-8) / (spec_mel_r.abs() + 1e-8))  # [M,T]

        # 3) Mel IPD
        IPD = spec_mel_l.angle() - spec_mel_r.angle()
        SI = IPD.sin()
        CI = IPD.cos()

        # 4) GCC
        # GCC = torch.fft.ifft(spec_l*spec_r.conj()/spec_l.abs()/spec_r.abs())
        # GCC = GCC[0:self.n_mels,:].abs()
        GCC = torch.fft.ifft(spec_mel_l * spec_mel_r.conj() / (spec_mel_l.abs()+1e-8) / (spec_mel_r.abs()+1e-8)).abs()

        feature = torch.cat([
            spec_mel_l_mag.unsqueeze(0),
            spec_mel_r_mag.unsqueeze(0),
            ILD.unsqueeze(0),
            SI.unsqueeze(0),
            CI.unsqueeze(0),
            GCC.unsqueeze(0)
        ], dim=0)  # => [4, M, T]
        return feature

