import torch
import torch.nn as nn
from torchvision.models import resnet18
# from torch.nn import functional as F
from torchvision.models.resnet import BasicBlock
from models.conformer.encoder import ConformerBlock

class ResNetConformerSELD(nn.Module):
    def __init__(self, in_channels=4, num_classes=12, num_conformer_layers=8):
        super().__init__()

        # 1. Conv2D: 输入 C → 24
        self.input_conv = nn.Conv2d(in_channels, 24, kernel_size=3, padding=1)

        # 2. ResNet18 前4层（修改输入通道）
        self.layer1 = nn.Sequential(*[
            BasicBlock(inplanes=24, planes=24),
            BasicBlock(inplanes=24, planes=24),
            # nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2)) # Frequency Pooling 1（我们用 avgpool 替代）→ 沿频率维度进行池化
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))  # Frequency Pooling 1（我们用 maxpool 替代）→ 沿频率维度进行池化
        ])

        self.layer2 = nn.Sequential(*[
            BasicBlock(inplanes=24, planes=48,downsample=nn.Sequential(nn.Conv2d(24,48,kernel_size=1,stride=1),nn.BatchNorm2d(48))), #24->48
            BasicBlock(inplanes=48, planes=48),
            # nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2))  # Frequency Pooling 1（我们用 avgpool 替代）→ 沿频率维度进行池化
        ])

        self.layer3 = nn.Sequential(*[
            BasicBlock(inplanes=48, planes=96,
                       downsample=nn.Sequential(nn.Conv2d(48, 96, kernel_size=1, stride=1), nn.BatchNorm2d(96))),# 48->96
            BasicBlock(inplanes=96, planes=96),
            # nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2))  # Frequency Pooling 1（我们用 avgpool 替代）→ 沿频率维度进行池化
        ])

        self.layer4 = nn.Sequential(*[
            BasicBlock(inplanes=96, planes=192,
                       downsample=nn.Sequential(nn.Conv2d(96, 192, kernel_size=1, stride=1), nn.BatchNorm2d(192))), # 96->192

            BasicBlock(inplanes=192, planes=192),
            # nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2))  # Frequency Pooling 1（我们用 avgpool 替代）→ 沿频率维度进行池化
        ])

        # 3. Time Pooling 2（我们用 avgpool 替代）→ 沿时间维度进行池化
        # self.time_pool2 = nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.time_pool2 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

        # 4. Conv2D: 192 → 256（输入为 layer4 输出通道）
        self.conv256 = nn.Conv2d(192, 256, kernel_size=3, padding=1)

        # 5. N层Conformer：输入形状变为 [B, T, 256]
        self.conformer = nn.Sequential(*[
            ConformerBlock(encoder_dim = 256,num_attention_heads = 8) for _ in range(num_conformer_layers)
        ])

        # 6. Time Pooling 3：再一次池化时间维度
        # self.time_pool3 = nn.AdaptiveAvgPool2d((50,None))
        # self.time_pool3 = nn.AdaptiveMaxPool2d((50, 256))
        # self.time_pool3 = nn.MaxPool2d((26, 1), stride=(2, 1),dilation=(1,1))
        self.time_pool3 = nn.MaxPool1d(kernel_size=6, stride=2, dilation=1)

        self.resize_layer = nn.Linear(8192,256)

        # 7. FC（此部分基于paper：《THE NERC-SLIP SYSTEM FOR STEREO SOUND EVENT LOCALIZATION AND
        # DETECTION IN REGULAR VIDEO CONTENT OF DCASE 2025 CHALLENGE》）
        self.fc_sed1 = nn.Linear(256, 256)
        self.leaky_relu = nn.LeakyReLU()
        self.fc_sed2 = nn.Linear(256, num_classes*2) #输出每个类别的logits
        self.tanh = nn.Tanh()
        # self.norm = nn.BatchNorm2d


        # self.fc_doa1 = nn.Linear(256, 256)
        # self.fc_doa2 = nn.Linear(256, num_classes) #输出每个类别的DOA

    def forward(self, x):
        # x: [B, C, T, F]
        x = self.input_conv(x)  # [B, 24, T, F]
        x = self.layer1(x)      # [B, 24, T, F/2]
        x = self.layer2(x)      # [B, 49, T, F/2]
        x = self.layer3(x)      # [B, 96, T, F/2]
        x = self.layer4(x)      # [B, 192, T, F/2]

        x = self.time_pool2(x)  # [B, 192, T/2, F/2]

        x = self.conv256(x)     # [B, 256, T/2, F/2]
        B, C, T, Ff = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, T, C*Ff)  # [B, T/2, D]

        x = self.resize_layer(x)

        x = self.conformer(x)   # [B, T/2, 256]

        x = x.permute(0, 2, 1)
        x = self.time_pool3(x)  # [B, 50, 256]
        x = x.permute(0, 2, 1)

        # 逐时间帧进行分类
        x_sed = self.fc_sed1(x)
        x_sed = self.leaky_relu(x_sed)
        x_sed = self.fc_sed2(x_sed)
        x_accdoa = self.tanh(x_sed)
        x_accdoa = x_accdoa.view(x_accdoa.shape[0],x_accdoa.shape[1],x_accdoa.shape[2]//2,2)
        # x_doa = self.fc_doa1(x)
        # x_doa = self.fc_doa2(x_doa)   # [B, T', num_classes]

        return x_accdoa
