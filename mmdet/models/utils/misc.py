# Copyright (c) OpenMMLab. All rights reserved.
import random

import numpy as np
import torch
from torch import nn
from torch.autograd import Function
from torch.nn import functional as F


class SigmoidGeometricMean(Function):
    """Forward and backward function of geometric mean of two sigmoid
    functions.

    This implementation with analytical gradient function substitutes
    the autograd function of (x.sigmoid() * y.sigmoid()).sqrt(). The
    original implementation incurs none during gradient backprapagation
    if both x and y are very small values.
    """

    @staticmethod
    def forward(ctx, x, y):
        x_sigmoid = x.sigmoid()
        y_sigmoid = y.sigmoid()
        z = (x_sigmoid * y_sigmoid).sqrt()
        ctx.save_for_backward(x_sigmoid, y_sigmoid, z)
        return z

    @staticmethod
    def backward(ctx, grad_output):
        x_sigmoid, y_sigmoid, z = ctx.saved_tensors
        grad_x = grad_output * z * (1 - x_sigmoid) / 2
        grad_y = grad_output * z * (1 - y_sigmoid) / 2
        return grad_x, grad_y


sigmoid_geometric_mean = SigmoidGeometricMean.apply


def interpolate_as(source, target, mode='bilinear', align_corners=False):
    """Interpolate the `source` to the shape of the `target`.

    The `source` must be a Tensor, but the `target` can be a Tensor or a
    np.ndarray with the shape (..., target_h, target_w).

    Args:
        source (Tensor): A 3D/4D Tensor with the shape (N, H, W) or
            (N, C, H, W).
        target (Tensor | np.ndarray): The interpolation target with the shape
            (..., target_h, target_w).
        mode (str): Algorithm used for interpolation. The options are the
            same as those in F.interpolate(). Default: ``'bilinear'``.
        align_corners (bool): The same as the argument in F.interpolate().

    Returns:
        Tensor: The interpolated source Tensor.
    """
    assert len(target.shape) >= 2

    def _interpolate_as(source, target, mode='bilinear', align_corners=False):
        """Interpolate the `source` (4D) to the shape of the `target`."""
        target_h, target_w = target.shape[-2:]
        source_h, source_w = source.shape[-2:]
        if target_h != source_h or target_w != source_w:
            source = F.interpolate(
                source,
                size=(target_h, target_w),
                mode=mode,
                align_corners=align_corners)
        return source

    if len(source.shape) == 3:
        source = source[:, None, :, :]
        source = _interpolate_as(source, target, mode, align_corners)
        return source[:, 0, :, :]
    else:
        return _interpolate_as(source, target, mode, align_corners)


def gaussian(x, mean, std):
    # x = np.exp(-(x-mean)**2/(2*std**2))/(np.sqrt(2*3.1415926)*std)
    # x = np.exp(-(x-mean)**2/(2*std**2))/(2.5066282532517663*std)
    x = torch.exp(-(x-mean)**2/(2*std**2))/(2.5066282532517663*std)
    return x


def myactivate(x, func='none', dim=-1):
    """在输入x的最后1通道上进行激活，根据输入的激活函数func
        >>> x = torch.randn(3, 8)
        >>> z = torch.linspace(-1, 1, 8)
        >>> (z*x.softmax(1)).sum(1, True)
        >>> (z*x.sigmoid()).sum(1, True)
        >>> (z*x.sigmoid()/x.sigmoid().sum(1, True)).sum(1, True)
        >>> x.softmax(1)
        tensor([[0.1230, 0.1390, 0.1647, 0.4332, 0.1400],
                [0.2634, 0.0663, 0.0420, 0.5475, 0.0808],
                [0.5478, 0.0464, 0.0863, 0.2291, 0.0905]])
        >>> x.sigmoid()
        tensor([[0.5441, 0.5742, 0.6151, 0.8078, 0.5760],
                [0.7528, 0.4340, 0.3270, 0.8636, 0.4829],
                [0.7681, 0.2191, 0.3428, 0.5808, 0.3537]])
        >>> x.sigmoid()/x.sigmoid().sum(1, True)
        tensor([[0.1746, 0.1842, 0.1973, 0.2591, 0.1848],
                [0.2632, 0.1517, 0.1143, 0.3019, 0.1688],
                [0.3392, 0.0968, 0.1514, 0.2565, 0.1562]])
        >>> x / x.sum(-1, True)
        tensor([[ 0.0659,  0.1113,  0.1744,  0.5344,  0.1140],
                [ 0.5852, -0.1395, -0.3794,  0.9697, -0.0360],
                [-1.1972,  1.2701,  0.6504, -0.3258,  0.6024]])
        >>> (x - x.min(-1, True)[0]) / (x.max(-1, True)[0] - x.min(-1, True)[0] + 1e-7)
        tensor([[1.0000, 0.2629, 0.9849, 0.0000, 0.3837, 0.6112, 0.6978, 0.6595],
                [0.5647, 0.0000, 1.0000, 0.7328, 0.5724, 0.6635, 0.6075, 0.3897],
                [0.0000, 0.7287, 0.6958, 0.6612, 1.0000, 0.8203, 0.6411, 0.0724]])
    """
    if func == 'none':
        x = x
    elif func == 'softmax':
        x = torch.softmax(x, dim)
    elif func == 'sigmoid':
        x = torch.sigmoid(x)
    elif func == 'sigmsum':
        x = torch.sigmoid(x)
        x = x / (x.sum(dim, keepdim=True) + 1e-8)
    elif func == 'uniform':
        x = x / (x.sum(dim, keepdim=True) + 1e-8)
    elif func == 'minform':
        x = x - x.min(dim, True)[0]
    elif func == 'maxform':
        x = x / (x.max(dim, True)[0] + 1e-8)
    elif func == 'minmax':
        x = (x - x.min(dim, True)[0]) / (x.max(dim, True)[0] - x.min(dim, True)[0] + 1e-8)
    elif func == 'minmaxsum':
        x = (x - x.min(dim, True)[0]) / (x.max(dim, True)[0] - x.min(dim, True)[0] + 1e-8)
        x = x / (x.sum(dim, keepdim=True) + 1e-8)
    else:
        raise NotImplementedError(f'func={func}')
    return x


class Integral(nn.Module):
    """A fixed layer for calculating integral result from distribution.
    This layer calculates the target location by :math: `sum{P(y_i) * y_i}`,
    P(y_i) denotes the softmax vector that represents the discrete distribution
    y_i denotes the discrete set, usually {0, 1, 2, ..., reg_max}
    Args:
        reg_max (int): The maximal value of the discrete set. Default: 16. You
            may want to reset it according to your new dataset or related
            settings.
    """
    def __init__(self, reg_val={'min':0, 'max':16, 'num':17, 'activate': 'softmax', 'method': 'v1', 'usedfl': False}):
        super(Integral, self).__init__()
        assert reg_val['method'] in ['v1', 'v2', 'v3']
        self.reg_min = reg_val['min']
        self.reg_max = reg_val['max']
        self.reg_num = reg_val['num']
        self.reg_act = reg_val.get('activate', 'softmax')
        self.reg_method = reg_val.get('method', 'v1')
        self.register_buffer('project', torch.linspace(self.reg_min, self.reg_max, self.reg_num))

    def forward(self, x, keepdim=True, **kwargs):
        if self.reg_method == 'v1':         # for yoloY
            return self.forward_v1(x, keepdim, **kwargs)
        elif self.reg_method == 'v2':       # for aaamixer
            return self.forward_v2(x, keepdim, **kwargs)
        elif self.reg_method == 'v3':       # for aaamixer
            return self.forward_v3(x, keepdim, **kwargs)
        else:
            raise NotImplementedError(f'self.reg_method = {self.reg_method}')

    def forward_v1(self, x, keepdim=True, **kwargs):
        """Forward feature from the regression head to get integral result of bounding box location.
        Args:
            x (Tensor): Features of the regression head, shape (N, 4*(n+1)), n is self.reg_max.
        Returns:
            x (Tensor): Integral result of box locations, i.e., distance
                offsets from the box center in four directions, shape (N, 4).
        """
        x = x.reshape(-1, self.reg_num)
        x = F.softmax(x, dim=1)
        x = F.linear(x, self.project.type_as(x))
        x = x if keepdim else x.reshape(-1, 4)
        return x

    def forward_v2(self, x, keepdim=True, **kwargs):
        """Forward feature from the regression head to get integral result of bounding box location.
        返回对delta_xyxy的预测：距离偏差的概率期望
        """
        N, n_query, n_pred = x.size()
        x = x.view(N, n_query, 4, self.reg_num)          # (N, n_query, 4*regnum) => (N, n_query, 4, regnum)
        x = myactivate(x, func=self.reg_act, dim=-1)

        # if random.random() > 0.998:
        #     stage = kwargs.pop('stage', None)
        #     print(f'\n====={stage}=======')
        #     delta_prob = x.sum(-1, True).view(N * n_query, 4)
        #     max_score, delta_label = x.max(dim=-1, keepdim=True)
        #     max_score = max_score.view(N * n_query, 4)
        #     delta_label = delta_label.view(N * n_query, 4)
        #     print('delta_prob => ', delta_prob[:3, :])
        #     print('max_score => ', max_score[:60, :])
        #     print('delta_label => ', delta_label[:60, :])

        x = (x * self.project.type_as(x)).sum(dim=-1)    # (N, n_query, 4, regnum) => (N, n_query, 4)
        # x = x if keepdim else x.view(-1, 4)

        # if random.random() > 0.9999:
        #     stage = kwargs.pop('stage', None)
        #     print(f'\n====={stage}=======')
        #     print('delta_xyxy => ', x[0, 50:60, :])
        return x

    def forward_v3(self, x, keepdim=True, **kwargs):
        """Forward feature from the regression head to get integral result of bounding box location.
        返回对delta_xyxy的预测：距离偏差的概率峰值
        """
        N, n_query, n_pred = x.size()
        x = x.view(N, n_query, 4, self.reg_num)          # (N, n_query, 4*regnum) => (N, n_query, 4, regnum)
        x = myactivate(x, func=self.reg_act, dim=-1)

        # delta_prob = x.sum(-1, True).view(N * n_query, 4)
        # max_score, delta_label = x.max(dim=-1, keepdim=True)
        # max_score = max_score.view(N * n_query, 4)
        # delta_label = delta_label.view(N * n_query, 4)
        # print('delta_prob => ', delta_prob[:, :])
        # print('delta_label => ', delta_label[:, :])

        project = self.project.type_as(x)
        project = project.view(1, 1, 1, self.reg_num).repeat(N, n_query, 4, 1)
        # index = x.max(dim=-1, keepdim=True)[1]
        project = project.gather(dim=-1, index=x.max(dim=-1, keepdim=True)[1])
        project = project.view(N, n_query, 4)
        # if random.random() > 1.999:
        #     stage = kwargs.pop('stage', None)
        #     print(f'\n====={stage}=======')
        #     print('delta_xyxy => \n', project[0, :5, :])
        return project





