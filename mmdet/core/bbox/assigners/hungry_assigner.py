# Copyright (c) OpenMMLab. All rights reserved.
import random

import mmcv
import torch
import torch.nn.functional as F

from ..builder import BBOX_ASSIGNERS
from ..match_costs import build_match_cost
from ..transforms import bbox_cxcywh_to_xyxy
from .assign_result import AssignResult
from .base_assigner import BaseAssigner

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None


@BBOX_ASSIGNERS.register_module()
class HungryAssigner(BaseAssigner):
    """Computes one-to-one matching between predictions and ground truth.
    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt
    Args:
        cls_weight (int | float, optional): The scale factor for classification cost. Default 1.0.
        bbox_weight (int | float, optional): The scale factor for regression L1 cost. Default 1.0.
        iou_weight (int | float, optional): The scale factor for regression iou cost. Default 1.0.
        iou_calculator (dict | optional): The config for the iou calculation Default type `BboxOverlaps2D`.
        iou_mode (str | optional): "iou" (intersection over union), "iof"
                (intersection over foreground), or "giou" (generalized intersection over union). Default "giou".
    """
    def __init__(self,
                 cls_cost=dict(type='ClassificationCost', weight=1.),
                 reg_cost=dict(type='BBoxL1Cost', weight=1.0),
                 iou_cost=dict(type='IoUCost', iou_mode='giou', weight=1.0),
                 dfl_cost=dict(type='CrossEntropyLossCost', use_sigmoid=True, weight=2.0)):
        self.cls_cost = build_match_cost(cls_cost)
        self.reg_cost = build_match_cost(reg_cost)
        self.iou_cost = build_match_cost(iou_cost)
        self.dfl_cost = build_match_cost(dfl_cost)

    def assign(self,
               bbox_pred,
               cls_pred,
               delta_pred,
               gt_bboxes,
               gt_labels,
               delta_label,
               img_meta,
               gt_bboxes_ignore=None,
               eps=1e-7):
        """Computes one-to-one matching based on the weighted costs.

        This method assign each query prediction to a ground truth or
        background. The `assigned_gt_inds` with -1 means don't care,
        0 means negative sample, and positive number is the index (1-based)
        of assigned gt.
        The assignment is done in the following steps, the order matters.

        1. assign every prediction to -1
        2. compute the weighted costs
        3. do Hungarian matching on CPU based on the costs
        4. assign all to 0 (background) first, then for each matched pair
           between predictions and gts, treat this prediction as foreground
           and assign the corresponding gt index (plus 1) to it.

        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. Shape
                [num_query, 4].
            cls_pred (Tensor): Predicted classification logits, shape
                [num_query, num_class].
            gt_bboxes (Tensor): Ground truth boxes with unnormalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).
            img_meta (dict): Meta information for current image.
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`. Default None.
            eps (int | float, optional): A value added to the denominator for
                numerical stability. Default 1e-7.

        Returns:
            :obj:`AssignResult`: The assigned result.
        """
        assert gt_bboxes_ignore is None, \
            'Only case when gt_bboxes_ignore is None is supported.'
        num_gts, num_bboxes = gt_bboxes.size(0), bbox_pred.size(0)

        # 1. assign -1 by default
        assigned_gt_inds = bbox_pred.new_full((num_bboxes, ), -1, dtype=torch.long)
        assigned_labels = bbox_pred.new_full((num_bboxes, ), -1, dtype=torch.long)
        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels)
        img_h, img_w, _ = img_meta['img_shape']
        factor = gt_bboxes.new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(0)

        # 2. compute the weighted costs
        # classification cost.
        cls_cost = self.cls_cost(cls_pred, gt_labels)
        # regression L1 cost
        normalize_gt_bboxes = gt_bboxes / factor
        reg_cost = self.reg_cost(bbox_pred, normalize_gt_bboxes)
        # regression iou cost, defaultly giou is used in official DETR.
        bboxes = bbox_cxcywh_to_xyxy(bbox_pred) * factor
        iou_cost = self.iou_cost(bboxes, gt_bboxes)

        # regression CE-Distribution cost
        if delta_pred is not None and delta_label is not None:
            # dfl_cost = self.dfl_cost(delta_pred, delta_label)
            delta_pred = delta_pred[:, None, :].repeat(1, num_gts, 1)
            delta_pred = delta_pred.view(num_bboxes, num_gts, 4, -1)        # =>(nquery, ngtbox, 4, regnum)
            reg_num = delta_pred.size(-1)
            if self.dfl_cost.__class__.__name__ == 'CrossEntropyLossCost':
                delta_pred = delta_pred.view(-1, reg_num)
                delta_pred = myactivate(delta_pred, func=self.dfl_cost.activate, dim=-1)
                if self.dfl_cost.use_sigmoid:
                    # label => one-hot label  # (nquery, ngtbox, 4) => (-1, regnum)
                    delta_label = F.one_hot(delta_label, reg_num).view(-1, reg_num)
                    dfl_cost = F.binary_cross_entropy_with_logits(
                        delta_pred, delta_label.float(), reduction='none').mean(dim=-1)    # 沿regnum求均值
                else:
                    delta_label = delta_label.view(-1)    # (nquery, ngtbox, 4) => (nquery*ngtbox*4,)
                    dfl_cost = F.cross_entropy(delta_pred, delta_label, reduction='none')
                dfl_cost = dfl_cost.view(num_bboxes, num_gts, 4).sum(dim=-1) * self.dfl_cost.weight
            elif self.dfl_cost.__class__.__name__ == 'FocalLossCost':
                delta_pred = delta_pred.view(-1, reg_num)   # (nquery, ngtbox, 4, regnum) => (nquery*ngtbox*4, regnum)
                delta_pred = myactivate(delta_pred, func=self.dfl_cost.activate, dim=-1)
                delta_label = delta_label.view(-1)                      # (nquery, ngtbox, 4) => (nquery*ngtbox*4, )
                # dfl_cost = self.dfl_cost(delta_pred, delta_label)       # => (nquery*ngtbox*4, nquery*ngtbox*4) XXX!!!
                # 先沿各个类别求和，每个点都需要OneHot匹配；再沿xyxy求和,每次匹配需要4个坐标点都同时以最小代价匹配
                delta_weight, reduction, avg_factor = None, 'mean', None
                dfl_cost = py_focal_cost_with_prob(delta_pred, delta_label, delta_weight,
                                                   self.dfl_cost.gamma, self.dfl_cost.alpha,
                                                   reduction, avg_factor)
                dfl_cost = dfl_cost.view(num_bboxes, num_gts, 4).sum(dim=-1) * self.dfl_cost.weight
                # if random.random() > 1.999:
                #     print(f'\ndfl_cost=\n{dfl_cost[:10, :]}')
            else:
                raise NotImplementedError(f'{self.dfl_cost.__class__.__name__}')
        else:
            dfl_cost = iou_cost * self.dfl_cost.weight * 0

        # weighted sum of above three costs
        cost = cls_cost + reg_cost + iou_cost + dfl_cost

        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" '
                              'to install scipy first.')
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(bbox_pred.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(bbox_pred.device)

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]
        result = AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_labels)
        return result


def py_focal_cost_with_prob(pred,
                            target,
                            weight=None,
                            gamma=2.0,
                            alpha=0.25,
                            reduction='none',
                            avg_factor=None):
    """PyTorch version of `Focal Loss <https://arxiv.org/abs/1708.02002>`_.
    Different from `py_sigmoid_focal_loss`, this function accepts probability
    as input.
         Forked From ./mmdet/models/losses/focal_loss.py
         py_focal_loss_with_prob()    修改了加权求和操作
    Args:
        pred (torch.Tensor): The prediction probability with shape (N, C) or (N, 4, C),
            C is the number of classes.
        target (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    num_classes = pred.size(-1)
    target = F.one_hot(target, num_classes=num_classes + 1)
    target = target[..., :num_classes].type_as(pred)
    pt = (1 - pred) * target + pred * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy(pred, target, reduction='none') * focal_weight
    # reduce_loss COST计算中一般不需要reduce
    if weight is not None:
        assert weight.ndim == loss.ndim
        loss = loss * weight
    if reduction == 'none':
        loss = loss
    elif reduction == 'sum':
        loss = loss.sum(dim=-1)
    elif reduction == 'mean':
        loss = loss.mean(dim=-1)
    return loss


def reduce_loss(loss, reduction):
    """Reduce loss as specified.
    forked from ./mmdet/models/losses/utils.py
    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


@mmcv.jit(derivate=True, coderize=True)
def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.
    forked from ./mmdet/models/losses/utils.py
    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Average factor when computing the mean of losses.
    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            # Avoid causing ZeroDivisionError when avg_factor is 0.0,
            # i.e., all labels of an image belong to ignore index.
            eps = torch.finfo(torch.float32).eps
            loss = loss.sum() / (avg_factor + eps)
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def _expand_onehot_labels(labels, label_weights, label_channels, ignore_index):
    """Expand onehot labels to match the size of prediction."""
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    valid_mask = (labels >= 0) & (labels != ignore_index)
    inds = torch.nonzero(
        valid_mask & (labels < label_channels), as_tuple=False)

    if inds.numel() > 0:
        bin_labels[inds, labels[inds]] = 1

    valid_mask = valid_mask.view(-1, 1).expand(labels.size(0),
                                               label_channels).float()
    if label_weights is None:
        bin_label_weights = valid_mask
    else:
        bin_label_weights = label_weights.view(-1, 1).repeat(1, label_channels)
        bin_label_weights *= valid_mask

    return bin_labels, bin_label_weights, valid_mask


# from mmdet.models.utils.misc import myactivate
def myactivate(x, func='none', dim=-1):
    """在输入x的最后1通道上进行激活，根据输入的激活函数func"""
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