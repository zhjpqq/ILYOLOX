import torch.nn as nn
from mmcv.cnn import ConvModule


class involution(nn.Module):

    def __init__(self,
                 channels,
                 kernel_size,
                 stride):
        super(involution, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.channels = channels
        reduction_ratio = 4
        self.group_channels = 16
        self.groups = self.channels // self.group_channels #512/16=32
        self.conv1 = ConvModule(
            in_channels=channels, #512
            out_channels=channels // reduction_ratio, #512/4=128
            kernel_size=1,
            conv_cfg=None,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU'))
        self.conv2 = ConvModule(
            in_channels=channels // reduction_ratio,
            out_channels=kernel_size**2 * self.groups, #3**2 * 32 = 288
            kernel_size=1,
            stride=1,
            conv_cfg=None,
            norm_cfg=None,
            act_cfg=None)
        if stride > 1:
            self.avgpool = nn.AvgPool2d(stride, stride)
        self.unfold = nn.Unfold(kernel_size, 1, (kernel_size-1)//2, stride)

    def forward(self, x):
        weight = self.conv2(self.conv1(x if self.stride == 1 else self.avgpool(x))) # x:[8,512,17,20], self.conv1:[8,128,17,20], self.conv2: [8,288,17,20]
        b, c, h, w = weight.shape # b:28, c:288, h:17, w:20
        weight = weight.view(b, self.groups, self.kernel_size**2, h, w).unsqueeze(2) # weight:[8, 32, 1, 9, 17, 20]
        out = self.unfold(x).view(b, self.groups, self.group_channels, self.kernel_size**2, h, w)#unfold:实现手动滑窗 # out:[8, 32, 16, 9, 17, 20]
        out = (weight * out).sum(dim=3).view(b, self.channels, h, w) # out:[8,512,17,20]
        return out
