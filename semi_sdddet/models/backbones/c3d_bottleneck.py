from typing import List
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmengine.model import BaseModule
import torch.nn as nn
import torch
import math

class C3D_Block(BaseModule):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1, res=True):
        super().__init__()
        self.res = res
        mid_channels = out_channels // 4
        self.conv1 = ConvModule(in_channels=in_channels,
                                out_channels=mid_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=kernel_size // 2,
                                groups=groups,
                                act_cfg=dict(type='SiLU', inplace=True),
                                norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                                bias=False
                                )
        self.conv2 = ConvModule(in_channels=mid_channels,
                                out_channels=mid_channels,
                                kernel_size=3,
                                stride=stride,
                                padding=2,
                                groups=groups,
                                dilation=2,
                                act_cfg=dict(type='SiLU', inplace=True),
                                norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                                bias=False
                                )
        self.conv3 = ConvModule(in_channels=mid_channels,
                                out_channels=mid_channels,
                                kernel_size=3,
                                stride=stride,
                                padding=4,
                                groups=groups,
                                dilation=4,
                                act_cfg=dict(type='SiLU', inplace=True),
                                norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                                bias=False
                                )
        self.conv4 = ConvModule(in_channels=mid_channels,
                                out_channels=mid_channels,
                                kernel_size=3,
                                stride=stride,
                                padding=8,
                                groups=groups,
                                dilation=8,
                                act_cfg=dict(type='SiLU', inplace=True),
                                norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                                bias=False
                                )

    def forward(self, x):
        if self.res:
            res = x
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x2 = torch.cat((x1, x2, x3, x4), 1)
        
        b, n, h, w = x2.data.size()
        b_n = b * n // 2
        y = x2.reshape(b_n, 2, h * w)
        y = y.permute(1, 0, 2)
        y = y.reshape(2, -1, n // 2, h, w)
        
        o = torch.cat((y[0], y[1]), 1)
        if self.res:
            return res + o
        return o
    
    
class C3D_Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, res=True) -> None:
        super().__init__()
        self.blocks = nn.ModuleList()
        self.num_blocks = num_blocks
        self.conv = ConvModule(in_channels, out_channels, kernel_size=1)
        for _ in range(num_blocks):
            self.blocks.append(
                C3D_Block(out_channels, out_channels, res=res))
            
    def forward(self, x):
        x = self.conv(x)
        for i in range(self.num_blocks):
            x = self.blocks[i](x)
        return x
            
        