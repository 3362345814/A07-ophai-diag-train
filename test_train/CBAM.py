import torch.nn as nn
import torch
import torch.nn.functional as f

print("************************通道注意力机制*********************")


# 通道注意力机制
class ChannelAttention(nn.Module):
    # 初始化，in_planes参数指定了输入特征图的通道数，ratio参数用于控制通道注意力机制中特征降维和升维过程中的压缩比率。默认值为8
    def __init__(self, in_planes, ratio=8):
        # 继承父类初始化方法
        super().__init__()
        # 全局最大池化 [b,c,h,w]==>[b,c,1,1]
        self.avg_pool = nn.AdaptiveAvgPool3d((4, 1, 1))  # C*H*W
        # 全局平均池化 [b,c,h,w]==>[b,c,1,1]
        self.max_pool = nn.AdaptiveMaxPool3d((4, 1, 1))  # C*H*W

        # 使用1x1x1卷积核代替全连接层进行特征降维
        self.fc1 = nn.Conv3d(in_planes, in_planes // ratio, 1, bias=False)
        # 激活函数
        self.relu1 = nn.ReLU()
        # 使用1x1x1卷积核进行特征升维
        self.fc2 = nn.Conv3d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 通过平均池化和最大池化后的特征进行卷积、激活、再卷积操作
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))  # 平均池化-->卷积-->RELu激活-->卷积
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))  # 最大池化-->卷积-->RELu激活-->卷积
        # 将两个输出相加
        out = avg_out + max_out
        # 应用Sigmoid函数
        return self.sigmoid(out)


# 创建ChannelAttention模型的实例，输入通道数为256
model = ChannelAttention(256)
# 打印模型结构
print(model)
# 创建一个形状为[2, 256, 4, 8, 8]的输入张量，所有元素初始化为1
inputs = torch.ones([2, 256, 4, 8, 8])  # 2是批次大小（batch size），256是通道数，4、8、8分别是深度、高度和宽度的维度
# 将输入张量传递给模型，并获取输出
outputs = model(inputs)
# 打印输入张量的形状
print(inputs.shape)  # [2, 256, 4, 8, 8]
# 打印输出张量的形状
print(outputs.shape)  # [2, 256, 4, 1, 1]

print("************************空间注意力机制*********************")


# 空间注意力机制
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()

        # 断言法：kernel_size必须为3或7
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        # 三元操作：如果kernel_size的值等于7，则padding被设置为3；否则（即kernel_size的值为3），padding被设置为1。
        padding = 3 if kernel_size == 7 else 1
        # 定义一个卷积层，输入通道数为2，输出通道数为1
        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # (N, C, H, W)，dim=1沿着通道维度C，计算张量的平均值和最大值
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # 将平均值和最大值在通道维度上拼接
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


model = SpatialAttention(kernel_size=7)
print(model)
inputs = torch.ones([2, 256, 4, 8, 8])
outputs = model(inputs)
print(inputs.shape)
print(outputs.shape)

print("************************CBAM模块*********************")


class CBAM_Block(nn.Module):
    def __init__(self, channel, ratio=8, kernel_size=3):
        super().__init__()
        # 初始化通道注意力模块
        self.channelAttention = ChannelAttention(channel, ratio=ratio)
        # 初始化空间注意力模块
        self.SpatialAttention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        # 应用通道注意力和空间注意力
        x = x * self.channelAttention(x)
        x = x * self.SpatialAttention(x)
        return x


model = CBAM_Block(256)
print(model)
inputs = torch.ones([2, 256, 4, 8, 8])
outputs = model(inputs)
print(inputs.shape)
print(outputs.shape)
