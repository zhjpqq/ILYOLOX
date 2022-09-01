# Copyright (c) OpenMMLab. All rights reserved.
import math
import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (ConvModule, DepthwiseSeparableConvModule, bias_init_with_prob)
from mmcv.ops.nms import batched_nms
from mmcv.runner import force_fp32

from mmdet.core import bbox_overlaps
from mmdet.core import (MlvlPointGenerator, bbox_xyxy_to_cxcywh,
                        build_assigner, build_sampler, multi_apply, build_bbox_coder)
from mmdet.models.losses.utils import weight_reduce_loss as reduce_loss

from ..builder import HEADS, build_loss
from .base_dense_head import BaseDenseHead
from .dense_test_mixins import BBoxTestMixin
from ..utils.misc import Integral, myactivate


@HEADS.register_module()
class YOLOYHead(BaseDenseHead, BBoxTestMixin):
    """YOLOYHead head used in `YOLOY <https://arxiv.org/abs/2107.08430>`_.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of hidden channels in stacking convs.
            Default: 256
        stacked_convs (int): Number of stacking convs of the head.
            Default: 2.
        strides (tuple): Downsample factor of each feature map.
        use_depthwise (bool): Whether to depthwise separable convolution in
            blocks. Default: False
        dcn_on_last_conv (bool): If true, use dcn in the last layer of
            towers. Default: False.
        conv_bias (bool | str): If specified as `auto`, it will be decided by
            the norm_cfg. Bias of conv will be set as True if `norm_cfg` is
            None, otherwise False. Default: "auto".
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer. Default: None.
        loss_cls (dict): Config of classification loss.
        loss_iou (dict): Config of localization loss.
        loss_obj (dict): Config of objectness loss.
        loss_box (dict): Config of L1 loss.
        train_cfg (dict): Training config of anchor head.
        test_cfg (dict): Testing config of anchor head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 stacked_convs=2,
                 strides=[8, 16, 32],
                 use_depthwise=False,
                 dcn_on_last_conv=False,
                 conv_bias='auto',
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='Swish'),
                 loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, reduction='sum', loss_weight=1.0),
                 loss_obj=dict(type='CrossEntropyLoss', use_sigmoid=True, reduction='sum', loss_weight=1.0),
                 loss_box=dict(type='L1Loss', reduction='sum', loss_weight=1.0),
                 loss_iou=dict(type='IoULoss', mode='square', eps=1e-16, reduction='sum', loss_weight=5.0),
                 loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25),
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=dict(
                     type='Kaiming',
                     layer='Conv2d',
                     a=math.sqrt(5),
                     distribution='uniform',
                     mode='fan_in',
                     nonlinearity='leaky_relu'),

                 # prototyoe learning
                 prototype='',
                 # loss_sempcenter=dict(type='KnowledgeDistillationKLDivLoss', loss_weight=10, T=2),
                 # loss_sempsamper=dict(type='KnowledgeDistillationKLDivLoss', loss_weight=10, T=2),
                 # loss_geoproto=dict(type='KnowledgeDistillationKLDivLoss', loss_weight=10, T=2),

                 # add for il
                 has_teacher=False,
                 which_index='keepindex + posnoprev',
                 loss_distill='v0',
                 alpha_distill={'total': 1, 'cls': 1, 'loc': 1},
                 active_score=False,
                 active_funct='sigmoid',
                 mixxed_score=False,
                 hybrid_score=False,
                 onehot_score=False,
                 target_type='hard',
                 cates_distill='',
                 locat_distill='',
                 feats_distill='',
                 reg_val={'min': 0, 'max': 16, 'num': 17, 'use_dfl': False},
                 loss_cd_soft=dict(type='KnowledgeDistillationKLDivLoss', loss_weight=10, T=2),
                 loss_cd_obj=dict(type='MSELoss', reduction='mean', loss_weight=1),
                 loss_ld_soft=dict(type='KnowledgeDistillationKLDivLoss', loss_weight=1, T=2),
                 loss_ld_box=dict(type='SmoothL1Loss', loss_weight=10, reduction='mean'),
                 loss_ld_iou=dict(type='KnowledgeDistillationKLDivLoss',loss_weight=0.25, T=10),
                 loss_fd=dict(type='KnowledgeDistillationKLDivLoss', loss_weight=10, T=2),
                 ):

        super().__init__(init_cfg=init_cfg)
        self.has_teacher = has_teacher
        self.reg_val = reg_val
        self.use_dfl = self.reg_val['use_dfl']
        self.reg_out_channels = 4 * self.reg_val['num'] if self.use_dfl else 4

        self.num_classes = num_classes
        self.cls_out_channels = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.use_depthwise = use_depthwise
        self.dcn_on_last_conv = dcn_on_last_conv
        assert conv_bias == 'auto' or isinstance(conv_bias, bool)
        self.conv_bias = conv_bias
        self.use_sigmoid_cls = True

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self.loss_cls = build_loss(loss_cls)
        self.loss_iou = build_loss(loss_iou)
        self.loss_obj = build_loss(loss_obj)
        self.use_l1 = False  # This flag will be modified by hooks.  # TODO????????
        self.loss_box = build_loss(loss_box)
        if self.use_dfl:
            self.distpoint_coder = build_bbox_coder(dict(type='DistancePointBBoxCoder', clip_border=True))
            self.integral = Integral(reg_val)
            self.loss_dfl = build_loss(loss_dfl)

        self.prior_generator = MlvlPointGenerator(strides, offset=0)

        self.test_cfg = test_cfg
        self.train_cfg = train_cfg

        self.sampling = False
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            # sampling=False so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)

        self.which_index = which_index
        self.loss_distill = loss_distill
        self.alpha_distill = alpha_distill
        self.active_score = active_score
        self.mixxed_score = mixxed_score
        self.hybrid_score = hybrid_score
        self.onehot_score = onehot_score
        self.target_type = target_type
        self.active_funct = active_funct
        self.cates_distill = cates_distill
        self.locat_distill = locat_distill
        self.feats_distill = feats_distill
        self.loss_cd_soft = build_loss(loss_cd_soft)
        self.loss_cd_obj = build_loss(loss_cd_obj)
        self.loss_ld_soft = build_loss(loss_ld_soft)
        self.loss_ld_box = build_loss(loss_ld_box)
        self.loss_ld_iou = build_loss(loss_ld_iou)
        self.loss_fd = build_loss(loss_fd) if feats_distill else None

        self.fp16_enabled = False
        self._init_layers()

    def _init_layers(self):
        self.multi_level_cls_convs = nn.ModuleList()
        self.multi_level_reg_convs = nn.ModuleList()
        self.multi_level_conv_cls = nn.ModuleList()
        self.multi_level_conv_reg = nn.ModuleList()
        self.multi_level_conv_obj = nn.ModuleList()
        for _ in self.strides:
            self.multi_level_cls_convs.append(self._build_stacked_convs())
            self.multi_level_reg_convs.append(self._build_stacked_convs())
            conv_cls, conv_reg, conv_obj = self._build_predictor()
            self.multi_level_conv_cls.append(conv_cls)
            self.multi_level_conv_reg.append(conv_reg)
            self.multi_level_conv_obj.append(conv_obj)

    def _build_stacked_convs(self):
        """Initialize conv layers of a single level head."""
        conv = DepthwiseSeparableConvModule \
            if self.use_depthwise else ConvModule
        stacked_convs = []
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            if self.dcn_on_last_conv and i == self.stacked_convs - 1:
                conv_cfg = dict(type='DCNv2')
            else:
                conv_cfg = self.conv_cfg
            stacked_convs.append(
                conv(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    bias=self.conv_bias))
        return nn.Sequential(*stacked_convs)

    def _build_predictor(self):
        """Initialize predictor layers of a single level head."""
        conv_cls = nn.Conv2d(self.feat_channels, self.cls_out_channels, 1)
        conv_reg = nn.Conv2d(self.feat_channels, self.reg_out_channels, 1)
        conv_obj = nn.Conv2d(self.feat_channels, 1, 1)
        return conv_cls, conv_reg, conv_obj

    def init_weights(self):
        super(YOLOYHead, self).init_weights()
        # Use prior in model initialization to improve stability
        bias_init = bias_init_with_prob(0.01)
        for conv_cls, conv_obj in zip(self.multi_level_conv_cls,
                                      self.multi_level_conv_obj):
            conv_cls.bias.data.fill_(bias_init)
            conv_obj.bias.data.fill_(bias_init)

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      teacher_info=None,
                      task_labels=None,
                      **kwargs):
        """
        forked form BaseDenseHead.
        """
        student_feat = x if self.has_teacher and self.feats_distill else None

        outs = self.forward(x)
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore,
                           student_feat=student_feat, teacher_info=teacher_info,
                           task_labels=task_labels)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(*outs, img_metas=img_metas, cfg=proposal_cfg)
            return losses, proposal_list

    def forward_single(self, x, cls_convs, reg_convs, conv_cls, conv_reg, conv_obj):
        """Forward feature of a single scale level."""

        cls_feat = cls_convs(x)
        reg_feat = reg_convs(x)

        cls_score = conv_cls(cls_feat)
        bbox_pred = conv_reg(reg_feat)
        objectness = conv_obj(reg_feat)

        return cls_score, bbox_pred, objectness

    def forward(self, feats):
        """Forward features from the upstream network.
        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is a 4D-tensor.
        Returns:
            tuple[Tensor]: A tuple of multi-level predication map, each is a
                4D-tensor of shape (batch_size, 5+num_classes, height, width).
        """
        results = multi_apply(self.forward_single, feats,
                              self.multi_level_cls_convs,
                              self.multi_level_reg_convs,
                              self.multi_level_conv_cls,
                              self.multi_level_conv_reg,
                              self.multi_level_conv_obj)
        return results

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'objectnesses'))
    def loss(self,
             cls_scores,
             bbox_preds,
             objectnesses,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None,
             student_feat=[],
             teacher_info={},
             task_labels={}):
        """Compute loss of the head.
        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_priors * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_priors * 4.
            objectnesses (list[Tensor], Optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, 1, H, W).
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
        """
        num_imgs = len(img_metas)
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
        mlvl_priors = self.prior_generator.grid_priors(featmap_sizes, cls_scores[0].device, with_stride=True)

        flatten_cls_preds = [
            cls_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.cls_out_channels) for cls_pred in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.reg_out_channels) for bbox_pred in bbox_preds
        ]
        flatten_objectness = [
            objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1) for objectness in objectnesses
        ]

        flatten_cls_preds = torch.cat(flatten_cls_preds, dim=1)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_objectness = torch.cat(flatten_objectness, dim=1)
        flatten_mlvl_priors = torch.cat(mlvl_priors)
        flatten_bbox_decode = self._bbox_decode(flatten_mlvl_priors, flatten_bbox_preds)

        # 合并 GT-Label-Boxes & Teacher-Label-Boxes
        if self.has_teacher and 'hard' in self.cates_distill:
            for i in range(len(img_metas)):
                gt_labels[i] = torch.cat([teacher_info['pred_labels'][i], gt_labels[i]], dim=0)
                gt_bboxes[i] = torch.cat([teacher_info['pred_bboxes'][i], gt_bboxes[i]], dim=0)

        (pos_masks, cls_targets, cls_labels, obj_targets, bbox_targets, l1_targets,
         pos_assigned_gt_inds, imgs_whwh, num_fg_imgs) = multi_apply(
             self._get_target_single,
             flatten_cls_preds.detach(),
             flatten_objectness.detach(),
             flatten_mlvl_priors.unsqueeze(0).repeat(num_imgs, 1, 1),
             flatten_bbox_decode.detach(), gt_bboxes, gt_labels, img_metas)

        pos_masks = torch.cat(pos_masks, 0)
        cls_targets = torch.cat(cls_targets, 0)   # = cls_score * objectness
        cls_labels = torch.cat(cls_labels, 0)
        obj_targets = torch.cat(obj_targets, 0)
        bbox_targets = torch.cat(bbox_targets, 0)
        l1_targets = torch.cat(l1_targets, 0)
        imgs_whwh = torch.cat(imgs_whwh, 0)
        num_pos = pos_masks.nonzero().numel()
        # assert num_pos == pos_masks.nonzero().flatten().numel()
        # assert num_pos == max(sum(num_fg_imgs), 1)
        posindex = pos_masks.nonzero()

        # # 去除PrevTask的损失计算
        # label_indexs = cls_targets.max(dim=1, keepdim=False)[1]
        # assert (label_indexs - cls_labels).sum() == 0, \
        #     print(f'iou-aware-labels=> {(label_indexs - cls_labels).sum()}')
        label_indexs = cls_labels
        tasks_indexs = label_indexs.new_ones(cls_targets.size())

        # # 验证标签分配结果
        # batch_assign_labels = []
        # for i, pag_index in enumerate(pos_assigned_gt_inds):
        #     labels = torch.Tensor([gt_labels[i][idx].item() for idx in pag_index])
        #     batch_assign_labels.append(labels.long().to(pag_index.device))
        # batch_gt_labels = torch.cat(gt_labels, dim=0)
        # batch_assign_labels = torch.cat(batch_assign_labels, dim=0)
        # assert (batch_assign_labels - label_indexs).sum() == 0, print(f'{batch_assign_labels - label_indexs}')
        # print(f'\nloss.gt_labels=> {batch_assign_labels.numel()}', batch_assign_labels - label_indexs)

        # 区分新旧样本
        if 'posnoprev' in self.which_index:
            for label in task_labels['prev']:
                tasks_indexs[label_indexs == label, ...] = 0
        old_index = (1 - tasks_indexs[:, 1]).bool()
        old_mask = pos_masks.new_zeros(size=pos_masks.size()).bool()
        old_mask[posindex[old_index]] = True
        # old_gt_nums = [x.numel() for x in pos_assigned_gt_inds]
        # old_assigned_gt_inds = pos_assigned_gt_inds[old_index]
        new_index = tasks_indexs[:, 1].bool()
        new_mask = pos_masks.new_zeros(size=pos_masks.size()).bool()
        new_mask[posindex[new_index]] = True
        avg_factor = new_mask.nonzero().numel()

        # 区分样本权值
        cls_weights, box_weights, obj_weights = None, None, None
        if 'labelwprev' in self.which_index:
            # label_targets = cls_targets.max(dim=1, keepdim=False)[1]
            cls_weights = label_indexs.new_ones(cls_targets.size())
            box_weights = bbox_targets.new_ones(bbox_targets.size())
            # obj_weights = obj_targets.new_ones(obj_targets.size())
            for label in task_labels['prev']:
                cls_weights[label_indexs == label, ...] = self.alpha_distill['cls']
                box_weights[label_indexs == label, ...] = self.alpha_distill['loc']
                # obj_weights[label_targets == label, ...] = 0
            cls_weights = cls_weights[new_index]
            box_weights = box_weights[new_index]

        loss_cls = self.loss_cls(flatten_cls_preds.view(-1, self.cls_out_channels)[new_mask], cls_targets[new_index], weight=cls_weights, avg_factor=avg_factor)
        loss_obj = self.loss_obj(flatten_objectness.view(-1, 1), obj_targets, weight=obj_weights, avg_factor=num_pos)
        # loss_obj = self.loss_obj(flatten_objectness.view(-1, 1)[new_mask], obj_targets[new_mask], weight=obj_weights, avg_factor=avg_factor)
        loss_box = self.loss_box(flatten_bbox_preds.view(-1, 4)[new_mask], l1_targets[new_index], weight=box_weights, avg_factor=avg_factor)
        loss_iou = self.loss_iou(flatten_bbox_decode.view(-1, 4)[new_mask], bbox_targets[new_index], weight=box_weights, avg_factor=avg_factor)
        loss_dict = dict(loss_cls=loss_cls, loss_iou=loss_iou, loss_obj=loss_obj)       # loss_box=loss_box,

        if self.has_teacher and teacher_info:
            if self.loss_distill == 'v0':
                loss_distill = dict()
            elif self.loss_distill == 'v1':
                loss_distill = self.loss_distill_v1(
                    flatten_cls_preds, flatten_bbox_preds, flatten_objectness, flatten_mlvl_priors, flatten_bbox_decode,
                    teacher_info, student_feat, old_mask, num_pos, num_imgs, imgs_whwh, img_metas)
            elif self.loss_distill == 'v2':
                loss_distill = self.loss_distill_v2(
                    flatten_cls_preds, flatten_bbox_preds, flatten_objectness, flatten_mlvl_priors, flatten_bbox_decode,
                    teacher_info, student_feat, old_mask, num_pos, num_imgs, imgs_whwh, img_metas)
            elif self.loss_distill == 'v3':
                loss_distill = self.loss_distill_v3(
                    flatten_cls_preds, flatten_bbox_preds, flatten_objectness, flatten_mlvl_priors, flatten_bbox_decode,
                    teacher_info, student_feat, cls_targets, old_mask, old_index, pos_assigned_gt_inds,
                    num_pos, num_imgs, imgs_whwh, img_metas, task_labels, gt_labels)
            else:
                raise NotImplementedError
            loss_dict.update(loss_distill)
        return loss_dict

    def loss_distill_v3(self, batch_cls_score, batch_box_score,
                        batch_objectness, batch_mlvl_priors, batch_box_decode,
                        teacher_info, student_feat, cls_targets, old_mask, old_index,
                        pos_assigned_gt_inds, num_pos, num_imgs, imgs_whwh, img_metas, task_labels, gt_labels):
        loss_dict = {}
        keep_mask = teacher_info['pred_keepid']
        old_num = old_index.nonzero().numel()
        reg_num = self.reg_val['num']
        imgs_whwh = imgs_whwh[old_mask]
        keepindex = keep_mask.nonzero()
        nums_per_img = [pred.size(0) for pred in teacher_info['pred_labels']]

        if old_num == 0:
            zero = imgs_whwh.new_tensor([0])
            loss_dict.update({'loss_cd_soft': zero}) if 'soft' in self.cates_distill else None
            loss_dict.update({'loss_cd_obj': zero}) if 'obj' in self.cates_distill else None
            loss_dict.update({'loss_ld_box': zero}) if 'box' in self.locat_distill else None
            loss_dict.update({'loss_ld_iou': zero}) if 'iou' in self.locat_distill else None
            loss_dict.update({'loss_fd':     zero}) if 'kldv' in self.feats_distill else None
            return loss_dict

        # Hard输出    Student & Teacher
        hard_cls_label = cls_targets[old_index].max(1)[1].float()
        teacher_gt_label = torch.cat(teacher_info['pred_labels'], 0)

        # 分类输出
        batch_cls_score = batch_cls_score.view(-1, self.cls_out_channels)[old_mask]
        soft_cls_score = [cls_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.cls_out_channels)
                         for cls_pred in teacher_info['head_outs'][0]]
        soft_cls_score = torch.cat(soft_cls_score, dim=1).view(-1, self.cls_out_channels)[keep_mask]

        # 物体输出
        batch_objectness = batch_objectness.view(-1, )[old_mask]
        soft_objectness = [obj_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1)
                           for obj_pred in teacher_info['head_outs'][2]]
        soft_objectness = torch.cat(soft_objectness, dim=1).view(-1, )[old_mask]
        hard_objectness = soft_objectness.new_ones(soft_objectness.size())

        # 定位输出
        batch_box_score = batch_box_score.view(-1, self.reg_out_channels)[old_mask]
        batch_box_decode = batch_box_decode.view(-1, 4)[old_mask]
        soft_box_score = [box_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.reg_out_channels)
                          for box_pred in teacher_info['head_outs'][1]]
        soft_box_score = torch.cat(soft_box_score, dim=1)
        soft_box_decode = self._bbox_decode(batch_mlvl_priors, soft_box_score)
        soft_box_score = soft_box_score.view(-1, self.reg_out_channels)[keep_mask]
        soft_box_decode = soft_box_decode.view(-1, self.reg_out_channels)[keep_mask]

        # Teacher & Student 一对多匹配
        start = 0
        hard_cls_target = []
        soft_cls_target = []
        soft_box_target = []
        for i in range(len(img_metas)):
            pag_index = pos_assigned_gt_inds[i]
            soft_score = soft_cls_score[start: start + nums_per_img[i], :]
            soft_debox = soft_box_decode[start: start + nums_per_img[i], :]
            for idx in pag_index:
                label = gt_labels[i][idx].item()
                # print(label, label in task_labels['prev'])
                if label in task_labels['prev']:
                    hard_cls_target.append(label)
                    soft_cls_target.append(soft_score[idx])
                    soft_box_target.append(soft_debox[idx])
                    # soft_label = soft_score[idx].max(0)[1]
                    # print(f'验证标签得分是否对应：{soft_label}-{label}-{soft_label==label}')
            start = start + nums_per_img[i]
        hard_cls_target = [soft_cls_score.new_tensor(hard_cls_target), cls_targets[old_index]][0]
        soft_cls_score = torch.cat(soft_cls_target, dim=0).view(-1, soft_cls_score.size(-1))
        soft_box_decode = torch.cat(soft_box_target, dim=0).view(-1, soft_box_decode.size(-1))

        # 取0比较，验证匹配结果与hard_cls_label一致！
        # try:
        #     print(f'ndistill.gt_labels=> {hard_cls_target.numel()} & {hard_cls_label.numel()} '
        #           f'不等数量={(hard_cls_target != hard_cls_label).nonzero().numel()} => '
        #           f'{hard_cls_target - hard_cls_label}')
        # except:
        #     pass
        # assert (hard_cls_target - hard_cls_label).sum() == 0, \
        #     print(f'ndistill.gt_labels=> {hard_cls_target - hard_cls_label}')
        # 验证Teancher&Student的Score/Decodebox输出一致
        # print('cls_score ==,!=', (batch_cls_score == soft_cls_score).nonzero().numel(),
        #       (batch_cls_score != soft_cls_score).nonzero().numel())
        # print('cls_score =>', batch_cls_score, soft_cls_score)

        # print('box_decode ==,!=', (batch_box_decode == soft_box_decode).nonzero().numel(),
        #       (batch_box_decode != soft_box_decode).nonzero().numel())
        # print('box_decode =>', batch_box_decode, soft_box_decode)
        # print('==,!=', (batch_objectness == soft_objectness).nonzero().numel(),
        #                (batch_objectness != soft_objectness).nonzero().numel())
        
        # 计算权重
        cls_weight, loc_weight, obj_weight, box_weight, iou_weight = [None] * 5
        if self.mixxed_score:
            # N阶协同多任务损失函数；+为0阶，相互包含N次为N阶；
            # 连续学习下的分类/定位协同蒸馏：加权系数在0~2之间，1处代表分类定位的平衡，
            # (cls, loc) = (坏,坏)，(坏,好)，(好,坏)，(好,好).
            cls_err = F.binary_cross_entropy_with_logits(batch_cls_score, soft_cls_score.sigmoid(), pos_weight=None, reduction='none')
            # obj_err = F.binary_cross_entropy_with_logits(batch_objectness, soft_objectness.sigmoid(), pos_weight=None, reduction='none')
            # obj_err = F.mse_loss(batch_objectness, soft_objectness, reduction='none')
            # cls_err = (cls_err / cls_err.max(-1, True)[0]).mean(-1)
            cls_err = cls_err.sum(-1)   # + obj_err             # TODO
            loc_err = 1 - bbox_overlaps(batch_box_decode, soft_box_decode, mode='giou', is_aligned=True, eps=1e-6)
            # assert cls_err.size() == loc_err.size()
            cls_weight = 2 * cls_err / (cls_err + loc_err)
            loc_weight = 2 * loc_err / (cls_err + loc_err)
            cls_weight = cls_weight.view(-1, 1).repeat(1, self.cls_out_channels)
            loc_weight = loc_weight.view(-1, 1).repeat(1, self.reg_out_channels)
            # print(f'cls_weight=>', cls_weight[:100])
            # print(f'loc_weight=>', loc_weight[:100])
            # loc_weight = None

        # 损失计算
        if 'soft' in self.cates_distill:
            avg_factor = old_num
            if self.loss_cd_soft._get_name() == 'KnowledgeDistillationKLDivLoss':
                # 默认 active_funct='none'  &  active_score=False
                batch_cls_score = myactivate(batch_cls_score, func='none', dim=-1)
                soft_cls_score = myactivate(soft_cls_score, func='none', dim=-1)
            elif self.loss_cd_soft._get_name() == 'CrossEntropyLoss':
                # 默认 active_funct= none & F.sigmoid  &  active_score=False
                batch_cls_score = myactivate(batch_cls_score, func='none', dim=-1)
                soft_cls_score = myactivate(soft_cls_score, func='sigmoid', dim=-1)
                # z = [soft_cls_score[i, :].cpu().numpy() for i in range(soft_cls_score.size(0))]
                if self.hybrid_score == 'hy1':
                    # x[F.one_hot(x.max(-1)[1], x.size(-1)).bool()].view(x.size()[:-1]) == x.max(-1)[0]
                    soft_cls_score[F.one_hot(soft_cls_score.max(-1)[1], self.num_classes).bool()] = 1.0
                    # soft_cls_score = soft_cls_score / soft_cls_score.max(-1, True)[0]
                elif self.hybrid_score == 'hy2':
                    soft_score_delta = 1 - soft_cls_score.max(-1)[0]
                    soft_cls_score = soft_cls_score + soft_score_delta.view(-1, 1).repeat(1, self.num_classes)
                elif self.hybrid_score == 'hy3':
                    cls_weight = (1 - soft_cls_score.max(-1)[0]).pow(2)
                    cls_weight = cls_weight.view(-1, 1).repeat(1, self.num_classes)
                if self.onehot_score:
                    soft_cls_score = F.one_hot(soft_cls_score.max(-1)[1], self.num_classes)
            elif self.loss_cd_soft._get_name() == 'FocalLoss':
                batch_cls_score = myactivate(batch_cls_score, func='sigmoid', dim=-1)
                soft_cls_score = myactivate(soft_cls_score, func='sigmoid', dim=-1)
            loss_cd_soft = self.loss_cd_soft(batch_cls_score, soft_cls_score,
                                             weight=cls_weight, avg_factor=avg_factor)
            loss_dict.update({'loss_cd_soft': loss_cd_soft * self.alpha_distill['cls']})

        if 'obj' in self.cates_distill:
            avg_factor = old_num
            loss_cd_obj = self.loss_cd_obj(batch_objectness, hard_objectness,
                                           weight=obj_weight, avg_factor=avg_factor)
            loss_dict.update({'loss_cd_obj': loss_cd_obj})

        if 'box' in self.locat_distill:
            avg_factor = old_num
            loss_ld_box = self.loss_ld_box(batch_box_decode / imgs_whwh, soft_box_decode / imgs_whwh,
                                           weight=loc_weight, avg_factor=avg_factor)
            loss_dict.update({'loss_ld_box': loss_ld_box * self.alpha_distill['loc']})

        if 'iou' in self.locat_distill:
            avg_factor = old_num
            loss_ld_iou = self.loss_ld_iou(batch_box_decode, soft_box_decode,
                                           weight=loc_weight, avg_factor=avg_factor)
            loss_dict.update({'loss_ld_iou': loss_ld_iou * self.alpha_distill['loc']})

        if 'kldv' in self.feats_distill:
            loss_fd = [self.loss_fd(sf, tf, weight=None, avg_factor=num_imgs)
                       for sf, tf in zip(student_feat, teacher_info['neck_feats'])]
            loss_dict.update({'loss_fd': loss_fd * self.alpha_distill['feat']})

        return loss_dict

    @torch.no_grad()
    def _get_target_single(self, cls_preds, objectness, priors, decoded_bboxes, gt_bboxes, gt_labels, img_meta):
        """Compute classification, regression, and objectness targets for priors in a single image.
        Args:
            cls_preds (Tensor): Classification predictions of one image,
                a 2D-Tensor with shape [num_priors, num_classes]
            objectness (Tensor): Objectness predictions of one image,
                a 1D-Tensor with shape [num_priors]
            priors (Tensor): All priors of one image, a 2D-Tensor with shape
                [num_priors, 4] in [cx, xy, stride_w, stride_y] format.
            decoded_bboxes (Tensor): Decoded bboxes predictions of one image,
                a 2D-Tensor with shape [num_priors, 4] in [tl_x, tl_y,
                br_x, br_y] format.
            gt_bboxes (Tensor): Ground truth bboxes of one image, a 2D-Tensor
                with shape [num_gts, 4] in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth labels of one image, a Tensor
                with shape [num_gts].
        """
        num_priors = priors.size(0)
        num_gts = gt_labels.size(0)
        gt_bboxes = gt_bboxes.to(decoded_bboxes.dtype)
        # No target
        if num_gts == 0:
            cls_target = cls_preds.new_zeros((0, self.num_classes))
            bbox_target = cls_preds.new_zeros((0, 4))
            l1_target = cls_preds.new_zeros((0, 4))
            obj_target = cls_preds.new_zeros((num_priors, 1))
            foreground_mask = cls_preds.new_zeros(num_priors).bool()
            return (foreground_mask, cls_target, obj_target, bbox_target, l1_target, None, 0)

        # YOLOY uses center priors with 0.5 offset to assign targets,
        # but use center priors without offset to regress bboxes.
        offset_priors = torch.cat([priors[:, :2] + priors[:, 2:] * 0.5, priors[:, 2:]], dim=-1)

        assign_result = self.assigner.assign(
            cls_preds.sigmoid() * objectness.unsqueeze(1).sigmoid(),
            offset_priors, decoded_bboxes, gt_bboxes, gt_labels)
        # assign_result.labels[assign_result.gt_inds.nonzero()]
        sampling_result = self.sampler.sample(assign_result, priors, gt_bboxes)
        pos_inds = sampling_result.pos_inds
        num_pos_per_img = pos_inds.size(0)

        h, w, _ = img_meta['img_shape']
        img_whwh = gt_bboxes.new_tensor([[w, h, w, h]])
        # img_whwh = img_whwh.repeat(num_pos_per_img, 1)
        img_whwh = img_whwh.repeat(num_priors, 1)

        pos_ious = assign_result.max_overlaps[pos_inds]
        # IOU aware classification score
        cls_target = F.one_hot(sampling_result.pos_gt_labels, self.num_classes)
        cls_target = cls_target * pos_ious.unsqueeze(-1)                # TODO TODO ?? 偶尔打乱原始标签最大值位置
        cls_labels = sampling_result.pos_gt_labels
        pos_assigned_gt_inds = sampling_result.pos_assigned_gt_inds

        # # # 验证标签分配逻辑
        # pagi = sampling_result.pos_assigned_gt_inds
        # sysasign_labels = cls_target.max(dim=1, keepdim=False)[1]
        # assigned_labels = torch.Tensor([gt_labels[x].item() for x in pagi]).long().to(pos_ious.device)
        # print('\nsamp.gt_labels=>', assigned_labels - sampling_result.pos_gt_labels)
        # print('sysa.gt_labels=>', assigned_labels - sysasign_labels)

        obj_target = torch.zeros_like(objectness).unsqueeze(-1)
        obj_target[pos_inds] = 1
        bbox_target = sampling_result.pos_gt_bboxes
        l1_target = cls_preds.new_zeros((num_pos_per_img, 4))

        # if self.use_l1: #org
        l1_target = self._get_l1_target(l1_target, bbox_target, priors[pos_inds])
        foreground_mask = torch.zeros_like(objectness).to(torch.bool)
        foreground_mask[pos_inds] = 1
        return (foreground_mask, cls_target, cls_labels, obj_target, bbox_target, l1_target,
                pos_assigned_gt_inds, img_whwh, num_pos_per_img)

    def _get_l1_target(self, l1_target, gt_bboxes, priors, eps=1e-8):
        """Convert gt bboxes to center offset and log width height."""
        gt_cxcywh = bbox_xyxy_to_cxcywh(gt_bboxes)
        l1_target[:, :2] = (gt_cxcywh[:, :2] - priors[:, :2]) / priors[:, 2:]
        l1_target[:, 2:] = torch.log(gt_cxcywh[:, 2:] / priors[:, 2:] + eps)
        return l1_target

    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   objectnesses,
                   img_metas=None,
                   cfg=None,
                   rescale=False,
                   with_nms=True,
                   need_keepid=False,
                   task_labels={}):
        """Transform network outputs of a batch into bbox results.
        Args:
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * 4, H, W).
            objectnesses (list[Tensor], Optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, 1, H, W).
            img_metas (list[dict], Optional): Image meta info. Default None.
            cfg (mmcv.Config, Optional): Test / postprocessing configuration,
                if None, test_cfg would be used.  Default None.
            rescale (bool): If True, return boxes in original image space.
                Default False.
            with_nms (bool): If True, do nms before return boxes.
                Default True.
        Returns:
            list[list[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of
                the corresponding box.
        """
        assert len(cls_scores) == len(bbox_preds) == len(objectnesses)
        cfg = self.test_cfg if cfg is None else cfg
        scale_factors = [img_meta['scale_factor'] for img_meta in img_metas]

        num_imgs = len(img_metas)
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
        mlvl_priors = self.prior_generator.grid_priors(featmap_sizes, cls_scores[0].device, with_stride=True)

        # flatten cls_scores, bbox_preds and objectness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.cls_out_channels) for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.reg_out_channels) for bbox_pred in bbox_preds
        ]
        flatten_objectness = [
            objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1) for objectness in objectnesses
        ]

        flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1)
        flatten_cls_scores_sigmoid = flatten_cls_scores.sigmoid()
        flatten_objectness = torch.cat(flatten_objectness, dim=1)
        flatten_objectness_sigmoid = flatten_objectness.sigmoid()
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_mlvl_priors = torch.cat(mlvl_priors)
        flatten_bbox_decode = self._bbox_decode(flatten_mlvl_priors, flatten_bbox_preds)
        if rescale:
            flatten_bbox_decode[..., :4] /= flatten_bbox_decode.new_tensor(scale_factors).unsqueeze(1)

        result_list = []
        for img_id in range(len(img_metas)):
            img_shape = img_metas[img_id]['img_shape']
            cls_scores = flatten_cls_scores_sigmoid[img_id]
            obj_scores = flatten_objectness_sigmoid[img_id]
            box_decode = flatten_bbox_decode[img_id]
            dets, labels, keep_mask, max_labels = \
                self._bboxes_nms(cls_scores, box_decode, obj_scores, cfg, img_shape, need_keepid, task_labels)
            if need_keepid:     # 取出未激活值
                # max_score, max_label = torch.max(cls_scores, 1)
                # cls_obj = max_score[keep_mask] * obj_scores[keep_mask]
                # cls_obj = org_score.sigmoid() * org_object.sigmoid()
                max_index = F.one_hot(max_labels, self.num_classes).bool()
                org_score = flatten_cls_scores[img_id][max_index][keep_mask]
                org_objnes = flatten_objectness[img_id][keep_mask]
                result_list.append((dets, labels, org_score, org_objnes, keep_mask))
            else:
                result_list.append((dets, labels))
        return result_list

    def _bbox_decode(self, priors, bbox_preds):

        if self.use_dfl:
            # bbox_preds==(Batch, H*W*1, 4*REG_MAX)  t\r\b\l-distance
            assert bbox_preds.dim() == 3, f'(Batch, H*W*Priors, 4*REG_MAX).DIM==3, but get {bbox_preds.dim()}'
            img_nums = bbox_preds.size(0)
            bbox_preds = self.integral(bbox_preds, reshape=True)
            # bbox_preds = bbox_preds.reshape(img_nums, -1, 4)
            prior_center = torch.cat([self.anchor_center(priors)]*img_nums, dim=0)
            decoded_bboxes = self.distpoint_coder.decode(prior_center, bbox_preds)  # tblr值=>x1y1x2y2值
            decoded_bboxes = decoded_bboxes.reshape(img_nums, -1, 4)
            return decoded_bboxes

        xys = (bbox_preds[..., :2] * priors[:, 2:]) + priors[:, :2]
        whs = bbox_preds[..., 2:].exp() * priors[:, 2:]

        tl_x = (xys[..., 0] - whs[..., 0] / 2)
        tl_y = (xys[..., 1] - whs[..., 1] / 2)
        br_x = (xys[..., 0] + whs[..., 0] / 2)
        br_y = (xys[..., 1] + whs[..., 1] / 2)

        decoded_bboxes = torch.stack([tl_x, tl_y, br_x, br_y], -1)
        return decoded_bboxes

    def anchor_center(self, anchors):
        """Get anchor centers from anchors.
        forked from gfl_head.py
        Args:
            anchors (Tensor): Anchor list with shape (N, 4), "xyxy" format.

        Returns:
            Tensor: Anchor centers with shape (N, 2), "xy" format.
        """
        anchors_cx = (anchors[..., 2] + anchors[..., 0]) / 2
        anchors_cy = (anchors[..., 3] + anchors[..., 1]) / 2
        return torch.stack([anchors_cx, anchors_cy], dim=-1)

    def _bboxes_nms(self, cls_scores, bboxes, score_factor, cfg, img_shape, need_keepid, task_labels={}):
        max_scores, max_labels = torch.max(cls_scores, 1)
        final_scores = max_scores * score_factor
        if need_keepid:   # 教师模型采样通道
            if cfg.method == 'scale' and cfg.score_thr_small > 0 and cfg.size_thr_small > 0:
                # 小目标自适应分类定位得分筛选
                areas_gap = cfg.size_thr_big * cfg.size_thr_big \
                            - cfg.size_thr_small * cfg.size_thr_small
                areas_thr = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0]) \
                            - cfg.size_thr_small * cfg.size_thr_small
                final_thr = areas_thr / areas_gap * (cfg.score_thr - cfg.score_thr_small) + cfg.score_thr_small
                # print('max-min-mean=>', final_thr.max().item(), final_thr.min().item(), final_thr.mean().item())
                # final_thr_sort = copy.deepcopy(final_thr.sort(descending=True)[0])
                # print('final_thr_sort=>', final_thr_sort[:30])
                valid_mask = final_scores >= final_thr
                raise NotImplementedError

            elif cfg.method == 'cls_mean_std':
                prev_labels = sorted(task_labels['prev'])
                prev_confis = {k: max_scores[max_labels == k] for k in prev_labels}
                prev_cofthr = {k: v.mean() + 1 * v.std() for k, v in prev_confis.items()}
                prev_cofthr = {k: round(v.item(), 4) for k, v in prev_cofthr.items()}
                print(f'prev_cofthr=> {prev_cofthr}')
                final_thr = max_scores.new_zeros(max_scores.size()).fill_(cfg.score_thr)
                for k, v in prev_cofthr.items():
                    final_thr[max_labels == k] = v
                valid_mask = final_scores >= final_thr

            elif cfg.method == 'all_mean_std':
                final_thr = max_scores.mean() + 2 * max_scores.std()
                # print(f'final_thr => {final_thr}, ==>{max_scores.mean()}, ==>{max_scores.std()}')
                valid_mask = final_scores >= final_thr

            elif cfg.method == 're_sampling':
                prev_labels = sorted(task_labels['prev'])
                # prev_indexs = {k: (max_labels == k).nonzero().flatten() for k in prev_labels}
                # prev_confis = {k: max_scores[max_labels == k] for k in prev_labels}
                # prev_bboxes = {k: bboxes[max_labels == k] for k in prev_labels}
                prev_indexs, prev_confis, prev_bboxes = {}, {}, {}
                for k in prev_labels:
                    prev_indexs.update({k: (max_labels == k).nonzero().flatten()})
                    prev_confis.update({k: max_scores[max_labels == k]})
                    prev_bboxes.update({k: bboxes[max_labels == k]})
                # prev_cofthr = {k: v.mean() + 1 * v.std() for k, v in prev_confis.items()}
                prev_okobjs = {k: prev_confis[k] >= cfg.score_thr_center for k in prev_labels}
                prev_okidxs = {k: prev_okobjs[k].nonzero().flatten() for k in prev_labels}
                # prev_boxious = {k: bbox_overlaps(v, v, mode='iou') for k, v in prev_bboxes.items()}
                # (prev_boxious[3].flatten() > 0.3).flatten().nonzero().numel()
                prev_okboxs = {k: prev_bboxes[k][prev_okidxs[k]] for k in prev_labels}
                prev_okious = {k: bbox_overlaps(prev_okboxs[k], prev_bboxes[k], mode='iou') for k in prev_labels}
                # (prev_okious[3] > 0.3).nonzero()[:, 1]
                prev_others = {k: (prev_okious[k] >= cfg.iou_thr_other).nonzero()[:, 1] for k in prev_labels}
                # prev_oknums = {k: prev_okobjs[k].nonzero().numel() for k in prev_labels}
                prev_okalls = copy.deepcopy(prev_okobjs)
                for k in prev_labels:
                    prev_okalls[k][prev_others[k].unique()] = True
                # prev_oknumx = {k: prev_okalls[k].nonzero().numel() for k in prev_labels}
                valid_mask = final_scores.new_zeros(size=final_scores.size()).bool()
                for k in prev_labels:
                    valid_mask[prev_indexs[k]] = prev_okalls[k]
                valid_mask = valid_mask & (final_scores >= cfg.score_thr_other)
                # print(f'prev_oknums={prev_oknums} \nprev_oknumx={prev_oknumx}')

            elif cfg.method == 'all_by_one':
                valid_mask = final_scores >= cfg.score_thr

            else:
                raise NotImplementedError(f'Unknown Branch')

        else:  # 学生模型测试通道
            valid_mask = final_scores >= cfg.score_thr

        bboxes = bboxes[valid_mask]
        scores = final_scores[valid_mask]
        labels = max_labels[valid_mask]
        keep_mask = valid_mask.new_zeros(size=valid_mask.size())
        nms_mask = valid_mask.new_zeros(size=scores.size())

        if labels.numel() == 0:
            dets, labels = bboxes, labels
        else:
            dets, keep = batched_nms(bboxes, scores, labels, cfg.nms)
            keep, paixu = keep.sort()
            dets = dets[paixu]
            labels = labels[keep]
            nms_mask[keep] = True
            keep_mask[valid_mask] = nms_mask
            # print(f'labels after nms => {labels.flatten().numel()}, '
            #       f'labels before nms => {max_labels[valid_mask].flatten().numel()}')

        return dets, labels, keep_mask, max_labels

    def loss_distill_v2(self, batch_cls_score, batch_box_score, batch_objectness, batch_mlvl_priors, batch_box_decode,
                        teacher_info, student_feat, pos_masks, num_pos, num_imgs, imgs_whwh, img_metas):
        loss_dict = {}
        if 'keepindex' in self.which_index:
            keepidx = teacher_info['pred_keepid']
        else:   # posindex
            keepidx = pos_masks
        keep_num = keepidx.nonzero().numel()
        reg_num = self.reg_val['num']
        cd_T = getattr(self.loss_cd_soft, 'T', None)
        ld_T = getattr(self.loss_ld_soft, 'T', None)
        imgs_whwh = imgs_whwh[keepidx]

        # 获得输出
        batch_cls_score = batch_cls_score.view(-1, self.cls_out_channels)[keepidx]
        soft_cls_score = [cls_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.cls_out_channels)
                          for cls_pred in teacher_info['head_outs'][0]]
        soft_cls_score = torch.cat(soft_cls_score, dim=1).view(-1, self.cls_out_channels)[keepidx]
        # print('==,!=', (batch_cls_score == soft_cls_score).nonzero().numel(),
        #                (batch_cls_score != soft_cls_score).nonzero().numel())

        batch_objectness = batch_objectness.view(-1, )[keepidx]
        soft_objectness = [obj_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1)
                           for obj_pred in teacher_info['head_outs'][2]]
        soft_objectness = torch.cat(soft_objectness, dim=1).view(-1, )[keepidx]

        batch_box_score = batch_box_score.view(-1, self.reg_out_channels)[keepidx]
        batch_box_decode = batch_box_decode.view(-1, 4)[keepidx]
        soft_box_score = [box_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.reg_out_channels)
                          for box_pred in teacher_info['head_outs'][1]]
        soft_box_score = torch.cat(soft_box_score, dim=1)
        soft_box_decode = self._bbox_decode(batch_mlvl_priors, soft_box_score)
        soft_box_score = soft_box_score.view(-1, self.reg_out_channels)[keepidx]
        soft_box_decode = soft_box_decode.view(-1, 4)[keepidx]
        # print('==,!=', (batch_box_score == soft_box_score).nonzero().numel(),
        #                (batch_box_score != soft_box_score).nonzero().numel())
        # print('==,!=', (batch_box_decode == soft_box_decode).nonzero().numel(),
        #                (batch_box_decode != soft_box_decode).nonzero().numel())

        # 输出激活
        if self.active_score:
            batch_cls_score = myactivate(batch_cls_score, func=self.active_funct, dim=-1)
            soft_cls_score = myactivate(soft_cls_score, func=self.active_funct, dim=-1)
        if self.use_dfl:
            batch_box_score = myactivate(batch_box_score, func=self.reg_val['activate'], dim=-1)
            soft_box_score = myactivate(soft_box_score, func=self.reg_val['activate'], dim=-1)

        # 输出调节
        if self.hybrid_score and keep_num > 0:
            # x[F.one_hot(x.max(-1)[1], x.size(-1)).bool()].view(x.size()[:-1]) == x.max(-1)[0]
            soft_cls_score[F.one_hot(soft_cls_score.max(-1)[1], self.num_classes).bool()] = 1.0
            if self.use_dfl:
                soft_box_score[F.one_hot(soft_box_score.max(-1)[1], self.reg_out_channels).bool()] = 1.0

        # 计算差值
        if 'soft' in self.cates_distill:
            if self.loss_cd_soft._get_name() == 'KnowledgeDistillationKLDivLoss':
                # 默认 active_funct=F.softmax  &  active_score=False
                batch_cls_score = myactivate(batch_cls_score/cd_T, func=self.active_funct, dim=-1)
                soft_cls_score = myactivate(soft_cls_score/cd_T, func=self.active_funct, dim=-1)
                cls_err = F.kl_div(
                    batch_cls_score.log(),
                    soft_cls_score.detach(),
                    reduction='none').mean(-1) * (cd_T * cd_T)
            elif self.loss_cd_soft._get_name() == 'CrossEntropyLoss':
                # 默认 active_funct=F.sigmoid  &  active_score=False
                batch_cls_score = myactivate(batch_cls_score, func=self.active_funct, dim=-1)
                soft_cls_score = myactivate(soft_cls_score, func=self.active_funct, dim=-1)
                cls_err = F.binary_cross_entropy(
                    batch_cls_score, soft_cls_score,
                    weight=None, reduction='none')
            elif self.loss_cd_soft._get_name() == 'FocalLoss':
                # forked from py_focal_loss_with_prob()
                pred = batch_cls_score
                target = soft_cls_score.type_as(pred)
                alpha, gamma = self.loss_cd_soft.alpha, self.loss_cd_soft.gamma
                pt = (1 - pred) * target + pred * (1 - target)
                focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * pt.pow(gamma)
                cls_err = F.binary_cross_entropy(pred, target, reduction='none') * focal_weight   #(keep_num, num_classes)
                cls_err = cls_err.sum(-1)
            elif self.loss_cd_soft._get_name() == 'MSELoss':
                cls_err = F.mse_loss(
                    batch_cls_score,
                    soft_cls_score,
                    reduction='none').sum(-1)
            else:
                raise NotImplementedError

        if 'objnes' in self.cates_distill:
            object_weight = None    # 忽略此损失计算
            obj_err = F.binary_cross_entropy(
                batch_objectness, soft_objectness,
                weight=object_weight, reduction='none')

        if 'soft' in self.locat_distill:
            if self.loss_ld_soft._get_name() == 'KnowledgeDistillationKLDivLoss':
                loc_err = F.kl_div(
                    (batch_box_score.view(keep_num, self.reg_out_channels) / ld_T).log(),
                    (soft_box_score.view(keep_num, self.reg_out_channels) / ld_T).detach(),
                    reduction='none').mean(-1) * (ld_T * ld_T)   # todo mean or sum
            elif self.loss_ld_soft._get_name() == 'FocalLoss':
                pred = batch_box_score.view(keep_num, 4, self.reg_out_channels // 4)
                target = soft_box_score.view(keep_num, 4, self.reg_out_channels // 4).type_as(pred)
                alpha, gamma = self.loss_ld_soft.alpha, self.loss_ld_soft.gamma
                pt = (1 - pred) * target + pred * (1 - target)
                focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * pt.pow(gamma)
                loc_err = F.binary_cross_entropy(pred, target, reduction='none') * focal_weight
                loc_err = loc_err.sum(-1).sum(-1)
            elif self.loss_ld_soft._get_name() == 'MSELoss':
                loc_err = F.mse_loss(
                    batch_box_score.view(keep_num, self.reg_out_channels),
                    soft_box_score.view(keep_num, self.reg_out_channels),
                    reduction='none').sum(-1)
            else:
                raise NotImplementedError

        l1dist = torch.abs(batch_box_decode / imgs_whwh - soft_box_decode / imgs_whwh)
        if self.loss_ld_box and self.loss_ld_box._get_name() == 'SmoothL1Loss':
            l1dist = torch.where(l1dist < 1.0, 0.5 * l1dist * l1dist / 1.0, l1dist - 0.5 * 1.0)
        xgiou = 1 - bbox_overlaps(batch_box_decode, soft_box_decode, mode='giou', is_aligned=True, eps=1e-6)

        # 计算损失权重
        cls_weight, loc_weight, bbox_weight, iou_weight = [None] * 4
        if self.mixxed_score and keep_num > 0:
            # N阶协同多任务损失函数；+为0阶，相互包含N次为N阶；
            # 连续学习下的分类/定位协同样本挖掘：共同样本挖掘：0~2之间，1处代表分类定位的平衡，
            # (cls, loc) = (坏,坏)，(坏,好)，(好,坏)，(好,好).
            cls_weight = 2 * cls_err / (cls_err + loc_err)
            loc_weight = 2 * loc_err / (cls_err + loc_err)

        # 计算损失
        if 'kldv' in self.feats_distill:
            # # assert len(student_feat) == len(teacher_info['neck_feats'])
            # print('sf != tf => ', [(sf != tf).nonzero().numel() for sf, tf in
            #                        zip(student_feat, teacher_info['neck_feats'])])
            # print('sf == tf => ', [(sf == tf).nonzero().numel() for sf, tf in
            #                        zip(student_feat, teacher_info['neck_feats'])])
            loss_fd = [self.loss_fd(sf, tf, weight=None, avg_factor=num_imgs)
                       for sf, tf in zip(student_feat, teacher_info['neck_feats'])]
            loss_dict.update({'loss_fd': loss_fd})

        if 'soft' in self.cates_distill:
            avg_factor = keep_num
            # loss_cd_soft = self.loss_cd_soft.loss_weight * reduce_loss(
            #                cls_err, cls_weight, self.loss_cd_soft.reduction, avg_factor)
            batch_cls_score = myactivate(batch_cls_score, func='none', dim=-1)
            soft_cls_score = myactivate(soft_cls_score, func='sigmoid', dim=-1)
            loss_cd_soft1 = self.loss_cd_soft(batch_cls_score, soft_cls_score,
                                              weight=cls_weight, avg_factor=avg_factor)
            # assert loss_cd_soft1 == loss_cd_soft
            loss_dict.update({'loss_cd_soft': loss_cd_soft1})

        if 'soft' in self.locat_distill:
            avg_factor = keep_num * 1
            loss_ld_soft = self.loss_ld_soft.loss_weight * reduce_loss(
                           loc_err, loc_weight, self.loss_ld_soft.reduction, avg_factor)
            # loss_ld_soft1 = self.loss_ld_soft(batch_box_score, soft_box_score,
            #                                   weight=loc_weight, avg_factor=avg_factor)
            # assert loss_ld_soft1 == loss_ld_soft
            loss_dict.update({'loss_ld_soft': loss_ld_soft})

        if 'box' in self.locat_distill:
            avg_factor = keep_num
            loss_ld_box = self.loss_ld_box.loss_weight * reduce_loss(
                           l1dist, bbox_weight, self.loss_ld_box.reduction, avg_factor)
            loss_ld_box1 = self.loss_ld_box(batch_box_decode / imgs_whwh, soft_box_decode / imgs_whwh,
                                            weight=bbox_weight, avg_factor=avg_factor)
            # assert loss_ld_box1 == loss_ld_box
            loss_dict.update({'loss_ld_box': loss_ld_box})

        if 'iou' in self.locat_distill:
            avg_factor = keep_num
            loss_ld_iou = self.loss_ld_iou.loss_weight * reduce_loss(
                          xgiou, iou_weight, self.loss_ld_iou.reduction, avg_factor)
            loss_ld_iou1 = self.loss_ld_iou(batch_box_decode, soft_box_decode,
                                            weight=iou_weight, avg_factor=avg_factor)
            # assert loss_ld_iou1 == loss_ld_iou
            loss_dict.update({'loss_ld_iou': loss_ld_iou})

        if keep_num == 0:
            assert all([x.item() == 0 for x in loss_dict.values()]), print(f'keep_num={keep_num}, {loss_dict}')

        return loss_dict

    def loss_distill_v1(self, flatten_cls_preds, flatten_bbox_preds,
                        flatten_objectness, flatten_mlvl_priors, flatten_bbox_decode,
                        teacher_info, student_feat, pos_masks, num_pos, num_imgs, imgs_whwh, img_metas):
        loss_dict = {}
        if 'keepindex' in self.which_index:
            keepidx = teacher_info['pred_keepid']
        else:
            keepidx = pos_masks
        keepnum = keepidx.numel()

        if 'soft' in self.cates_distill and False:
            batch_cls_preds = flatten_cls_preds.view(-1, self.cls_out_channels)[keepidx]
            soft_cls_preds = [cls_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.cls_out_channels)
                              for cls_pred in teacher_info['head_outs'][0]]
            soft_cls_preds = torch.cat(soft_cls_preds, dim=1).view(-1, self.cls_out_channels)[keepidx]
            if self.hybrid_score and keepnum > 0:
                # x[F.one_hot(x.max(-1)[1], x.size(-1)).bool()].view(x.size()[:-1]) == x.max(-1)[0]
                soft_cls_preds[F.one_hot(soft_cls_preds.max(-1)[1], self.num_classes).bool()] = 1.0
            loss_cd_soft = self.loss_cd_soft(batch_cls_preds, soft_cls_preds,
                                             weight=None, avg_factor=keepnum)
            loss_dict.update({'loss_cd_soft': loss_cd_soft})

        if 'soft2' in self.cates_distill:
            batch_cls_preds = flatten_cls_preds.view(-1, self.cls_out_channels)[keepidx]
            soft_cls_preds = [cls_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.cls_out_channels)
                              for cls_pred in teacher_info['head_outs'][0]]
            soft_cls_preds = torch.cat(soft_cls_preds, dim=1).view(-1, self.cls_out_channels)[keepidx]

            batch_cls_preds = myactivate(batch_cls_preds, func=self.active_score, dim=-1)
            soft_cls_preds = myactivate(soft_cls_preds, func=self.active_score, dim=-1)

            if self.hybrid_score:
                # x[F.one_hot(x.max(-1)[1], x.size(-1)).bool()].view(x.size()[:-1]) == x.max(-1)[0]
                soft_cls_preds[F.one_hot(soft_cls_preds.max(-1)[1], self.num_classes).bool()] = 1.0

            if self.loss_cd_soft._get_name() == 'KnowledgeDistillationKLDivLoss':
                cd_T = getattr(self.loss_cd_soft, 'T', None)
                cls_err = F.kl_div(
                    (batch_cls_preds / cd_T).log(),
                    (soft_cls_preds / cd_T).detach(),
                    reduction='none').mean(-1) * (cd_T * cd_T)
            elif self.loss_cd_soft._get_name() == 'FocalLoss':
                # forked from py_focal_loss_with_prob()
                pred = batch_cls_preds
                target = soft_cls_preds.type_as(pred)
                alpha, gamma = self.loss_cd_soft.alpha, self.loss_cd_soft.gamma
                pt = (1 - pred) * target + pred * (1 - target)
                focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * pt.pow(gamma)
                cls_err = F.binary_cross_entropy(pred, target, reduction='none') * focal_weight
                cls_err = cls_err.sum(-1)  # (keepnum, num_classes) => (keepnum, )
            elif self.loss_cd_soft._get_name() == 'MSELoss':
                cls_err = F.mse_loss(batch_cls_preds, soft_cls_preds, reduction='none')
                cls_err = cls_err.sum(-1)
            else:
                raise NotImplementedError
            avg_factor = keepnum
            cls_weight = None
            loss_cd_soft = self.loss_cd_soft.loss_weight * reduce_loss(
                cls_err, cls_weight, self.loss_cd_soft.reduction, avg_factor)
            # loss_cd_soft1 = self.loss_cd_soft(batch_cls_preds, soft_cls_preds,
            #                                   weight=cls_weight, avg_factor=avg_factor)
            # assert loss_cd_soft1 == loss_cd_soft
            loss_dict.update({'loss_cd_soft': loss_cd_soft})

        if 'bbox' in self.locat_distill:
            batch_soft_bbox = [box_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.reg_out_channels)
                               for box_pred in teacher_info['head_outs'][1]]
            batch_soft_bbox = torch.cat(batch_soft_bbox, dim=1)
            if 'decode' in self.locat_distill:
                batch_soft_bbox = self._bbox_decode(flatten_mlvl_priors, batch_soft_bbox)
                batch_pred_bbox = flatten_bbox_decode
            else:
                batch_pred_bbox = flatten_bbox_preds
            loss_ld_box = self.loss_ld_box(batch_pred_bbox.view(-1, 4)[keepidx],
                                             batch_soft_bbox.view(-1, 4)[keepidx],
                                             weight=None, avg_factor=keepnum)
            batch_soft_bbox = None
            loss_dict.update({'loss_ld_box': loss_ld_box})

        if 'logit' in self.locat_distill:
            # forked from ld_head.py Line99-Line122
            # batch_pred_bbox = flatten_bbox_preds.view(-1, self.reg_out_channels)[keepidx]
            batch_soft_bbox = [box_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.reg_out_channels)
                               for box_pred in teacher_info['head_outs'][1]]
            batch_soft_bbox = torch.cat(batch_soft_bbox, dim=1)
            loss_ld_iou = self.loss_ld_iou(flatten_bbox_preds.view(-1, self.reg_out_channels)[keepidx],
                                           batch_soft_bbox.view(-1, self.reg_out_channels)[keepidx],
                                           weight=None, avg_factor=keepnum)
            batch_soft_bbox = None
            loss_dict.update({'loss_ld_iou': loss_ld_iou})

        if 'kldv' in self.feats_distill:
            # assert len(student_feat) == len(teacher_info['neck_feats'])
            loss_fd = [self.loss_fd(sf, tf, weight=None, avg_factor=None)
                       for sf, tf in zip(student_feat, teacher_info['neck_feats'])]
            avg_factor = [1, len(loss_fd), len(img_metas), num_pos, ][2]
            loss_fd = sum(loss_fd) / avg_factor
            loss_dict.update({'loss_fd': loss_fd})

        return loss_dict
