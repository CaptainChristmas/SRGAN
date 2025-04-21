import math
import torch
from torch import nn

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, in_channels, num_heads=8):
        super(MultiHeadAttentionBlock, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(in_channels, num_heads)
        self.norm = nn.LayerNorm(in_channels)

    def forward(self, x):
        # 调整输入形状以适应 MultiheadAttention 的输入要求
        batch_size, channels, height, width = x.size()
        # 将特征图展平为序列
        x_flatten = x.view(batch_size, channels, -1).permute(2, 0, 1)  # [seq_len, batch_size, embed_dim]
        attn_output, _ = self.multihead_attn(x_flatten, x_flatten, x_flatten)
        attn_output = attn_output.permute(1, 2, 0).view(batch_size, channels, height, width)
        x = x_flatten.permute(1, 2, 0).view(batch_size, channels, height, width)

        # 在归一化之前将特征图展平
        x_flat = x.view(batch_size, channels, -1).permute(0, 2, 1)  # [batch_size, seq_len, channels]
        attn_output_flat = attn_output.view(batch_size, channels, -1).permute(0, 2, 1)  # [batch_size, seq_len, channels]
        norm_input = x_flat + attn_output_flat

        # 进行层归一化
        norm_output = self.norm(norm_input)

        # 恢复原来的形状
        x = norm_output.permute(0, 2, 1).view(batch_size, channels, height, width)

        return x

class Generator(nn.Module):
    def __init__(self, scale_factor):
        upsample_block_num = int(math.log(scale_factor, 2))

        super(Generator, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.attention_block = MultiHeadAttentionBlock(64)  # 引入多头注意力模块
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        block8 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(64, 3, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        attn_output = self.attention_block(block6)  # 使用多头注意力模块
        block7 = self.block7(attn_output)
        block8 = self.block8(block1 + block7)

        return (torch.tanh(block8) + 1) / 2
        
    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        attn_output = self.attention_block(block6)  # 使用多头注意力模块
        block7 = self.block7(attn_output)
        block8 = self.block8(block1 + block7)

        return (torch.tanh(block8) + 1) / 2

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # 基础层，不同尺度共享
        self.base_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )

        # 不同尺度的后续层
        self.scale_1_layers = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )
        self.scale_2_layers = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(128, 256, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, kernel_size=1)
        )

    def forward(self, x):
        # 基础层处理
        base_output = self.base_layers(x)

        # 尺度 1 处理
        scale_1_output = self.scale_1_layers(base_output)
        scale_1_output = torch.sigmoid(scale_1_output.view(x.size(0)))

        # 尺度 2 处理，这里可以对输入进行下采样，例如使用 nn.AvgPool2d
        downsampled_x = nn.AvgPool2d(kernel_size=2)(base_output)
        scale_2_output = self.scale_2_layers(downsampled_x)
        scale_2_output = torch.sigmoid(scale_2_output.view(x.size(0)))

        # 合并两个尺度的输出
        combined_output = (scale_1_output + scale_2_output) / 2
        return combined_output

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        # 使用分组卷积
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=4)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=4)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        return x + residual


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x
