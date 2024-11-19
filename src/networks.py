import torch
import torch.nn as nn
from kpn.network import KernelConv
import kpn.utils as kpn_utils
import numpy as np

# 定义基础网络类，包含权重初始化功能
class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    # 初始化网络权重
    def init_weights(self, init_type='normal', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            # 针对卷积层和线性层进行权重初始化
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)
                
                # 对偏置进行初始化为0
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            # 针对BatchNorm层的权重和偏置初始化
            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        # 遍历网络层并应用初始化函数
        self.apply(init_func)

# 定义图像修复生成器网络
class InpaintGenerator(BaseNetwork):
    def __init__(self, config=None, residual_blocks=8, init_weights=True):
        super(InpaintGenerator, self).__init__()

        # 配置滤波类型和卷积核大小
        self.filter_type = config.FILTER_TYPE
        self.kernel_size = config.kernel_size

        # 编码器的第一个卷积层
        self.encoder0 = nn.Sequential(
            nn.ReflectionPad2d(3),  # 反射填充，避免边界效应
            nn.Conv2d(in_channels=4, out_channels=64, kernel_size=7, padding=0),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True)
        )

        # 编码器的第二层卷积
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True)
        )

        # 编码器的第三层卷积
        self.encoder2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True)
        )

        # 中间的残差块部分
        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(256, 2)
            blocks.append(block)
        self.middle = nn.Sequential(*blocks)

        # 解码器部分，用于恢复图像尺寸
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, padding=0),
        )

        # 动态卷积操作模块
        self.kernel_pred = KernelConv(kernel_size=[3], sep_conv=False, core_bias=False)

        # 内核预测分支，生成动态卷积核
        self.kpn_model = kpn_utils.create_generator()

        # 初始化权重
        if init_weights:
            self.init_weights()

    def forward(self, x):
        # 输入图像的备份
        inputs = x.clone()

        # 编码阶段
        x = self.encoder0(x)  # 输出大小为 64*256*256
        x = self.encoder1(x)  # 输出大小为 128*128*128

        # 获取卷积核（图像层和特征层卷积核）
        kernels, kernels_img = self.kpn_model(inputs, x)

        # 进一步编码
        x = self.encoder2(x)  # 输出大小为 256*64*64

        # 应用动态卷积核对特征图进行过滤
        x = self.kernel_pred(x, kernels, white_level=1.0, rate=1)

        # 中间残差块处理
        x = self.middle(x)  # 输出大小为 256*64*64

        # 解码阶段
        x = self.decoder(x)  # 输出大小为 3*256*256

        # 应用动态卷积核对解码后图像进行细化处理
        x = self.kernel_pred(x, kernels_img, white_level=1.0, rate=1)

        # 使用tanh函数将输出映射到[0, 1]区间
        x = (torch.tanh(x) + 1) / 2

        return x

    # 保存特征图用于可视化分析
    def save_feature(self, x, name):
        x = x.cpu().numpy()
        np.save('./result/{}'.format(name), x)

# 判别器网络，用于区分真实图像和修复图像
class Discriminator(BaseNetwork):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True, init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid

        # 判别器的卷积层
        self.conv1 = self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        # 是否使用sigmoid激活函数
        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs, [conv1, conv2, conv3, conv4, conv5]

# 定义ResNet残差块
class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1, use_spectral_norm=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=dilation, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
        )

    def forward(self, x):
        # 残差连接
        out = x + self.conv_block(x)
        return out
#网络层的权重矩阵进行分解，通过最大奇异值对权重进行归一化
def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module
