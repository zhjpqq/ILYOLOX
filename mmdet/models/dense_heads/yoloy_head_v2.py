# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (ConvModule, DepthwiseSeparableConvModule, bias_init_with_prob)
from mmcv.ops.nms import batched_nms
from mmcv.runner import force_fp32

from mmdet.models.losses.utils import weight_reduce_loss as reduce_loss
from mmdet.core import bbox_overlaps
from mmdet.core import (MlvlPointGenerator, bbox_xyxy_to_cxcywh,
                        build_assigner, build_sampler, multi_apply, build_bbox_coder)
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
                 teacher_test_cfg=None,
                 cross_xfl=False,
                 learn_fgbg='foreback',
                 hybrid_score=False,
                 mixed_clsloc=False,
                 normal_clsscore=False,
                 which_index='keepindex + posnoprev',
                 alpha_distill=1.0,
                 loss_distill='v1',
                 cates_distill='',
                 locat_distill='',
                 feats_distill='',
                 # (posindex | keepindex) + (posandkeep | posnokeep | posnoprev | posonly)
                 reg_val={'min': 0, 'max': 16, 'num': 17, 'usedfl': False},
                 loss_cd_soft=dict(type='KnowledgeDistillationKLDivLoss', loss_weight=10, T=2),
                 loss_ld_soft=dict(type='KnowledgeDistillationKLDivLoss',loss_weight=0.25, T=10),
                 loss_ld_bbox=dict(type='SmoothL1Loss', loss_weight=10, reduction='mean'),
                 loss_ld_iou=dict(type='GIoULoss', loss_weight=2.0, reduction='mean'),
                 loss_fd=dict(type='KnowledgeDistillationKLDivLoss', loss_weight=10, T=2)):

        super().__init__(init_cfg=init_cfg)
        self.has_teacher = has_teacher
        self.teacher_test_cfg = teacher_test_cfg
        self.reg_val = reg_val
        self.use_dfl = self.reg_val['usedfl']
        self.reg_out_channels = 4 * self.reg_val['num'] if self.use_dfl else 4
        self.prototype = prototype
        self.cross_xfl = cross_xfl
        self.learn_fgbg = learn_fgbg
        self.hybrid_score = hybrid_score
        self.mixed_clsloc = mixed_clsloc
        self.normal_clsscore = normal_clsscore
        self.which_index = which_index
        self.alpha_distill = alpha_distill

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

        self.use_l1 = False  # This flag will be modified by hooks.
        self.loss_box = build_loss(loss_box)

        self.prior_generator = MlvlPointGenerator(strides, offset=0)

        self.test_cfg = test_cfg
        self.train_cfg = train_cfg

        self.sampling = False
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            # sampling=False so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)

        if self.use_dfl:
            self.integral = Integral(reg_val)
            self.loss_dfl = build_loss(loss_dfl)
            self.distpoint_coder = build_bbox_coder(dict(type='DistancePointBBoxCoder', clip_border=True))

        self.loss_distill = loss_distill
        self.cates_distill = cates_distill
        self.locat_distill = locat_distill
        self.feats_distill = feats_distill
        self.loss_cd_soft = build_loss(loss_cd_soft) #if cates_distill else None
        self.loss_ld_soft = build_loss(loss_ld_soft) #if 'soft' in locat_distill else None
        self.loss_ld_bbox = build_loss(loss_ld_bbox) #if 'box' in locat_distill else None
        self.loss_ld_iou = build_loss(loss_ld_iou) #if 'iou' in locat_distill else None
        self.loss_fd = build_loss(loss_fd) #if feats_distill else None

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
                      teacher_feat=None,
                      teacher_info={},
                      task_labels={},
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
                           student_feat=student_feat, teacher_info=teacher_info, task_labels=task_labels)
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
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
        Returns:
            tuple[Tensor]: A tuple of multi-level predication map, each is a
                4D-tensor of shape (batch_size, 5+num_classes, height, width).
        """
        return multi_apply(self.forward_single, feats,
                           self.multi_level_cls_convs,
                           self.multi_level_reg_convs,
                           self.multi_level_conv_cls,
                           self.multi_level_conv_reg,
                           self.multi_level_conv_obj)

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

        (pos_masks, cls_targets, obj_targets, bbox_targets, l1_targets, imgs_whwh, num_fg_imgs) = multi_apply(
             self._get_target_single,
             flatten_cls_preds.detach(),
             flatten_objectness.detach(),
             flatten_mlvl_priors.unsqueeze(0).repeat(num_imgs, 1, 1),
             flatten_bbox_decode.detach(), gt_bboxes, gt_labels, img_metas)

        # num_pos = max(sum(num_fg_imgs), 1)    # org
        pos_masks = torch.cat(pos_masks, 0)
        cls_targets = torch.cat(cls_targets, 0)   # maxVal = 1 * objectness
        obj_targets = torch.cat(obj_targets, 0)
        bbox_targets = torch.cat(bbox_targets, 0)
        l1_targets = torch.cat(l1_targets, 0)
        imgs_whwh = torch.cat(imgs_whwh, dim=0)
        num_pos = pos_masks.nonzero().numel()
        assert num_pos == max(sum(num_fg_imgs), 1)

        # 去除PrevTask的损失计算
        label_targets = cls_targets.max(dim=1, keepdim=False)[1]
        cls_weights = label_targets.new_ones(cls_targets.size())
        bbox_weights = bbox_targets.new_ones(bbox_targets.size())
        if 'posnoprev' in self.which_index:
            for label in task_labels['prev']:
                cls_weights[label_targets == label, ...] = 0
                bbox_weights[label_targets == label, ...] = 0

        loss_cls = self.loss_cls(flatten_cls_preds.view(-1, self.cls_out_channels)[pos_masks], cls_targets, weight=cls_weights, avg_factor=num_pos)
        loss_obj = self.loss_obj(flatten_objectness.view(-1, 1), obj_targets, weight=None, avg_factor=num_pos)
        loss_iou = self.loss_iou(flatten_bbox_decode.view(-1, 4)[pos_masks], bbox_targets, weight=bbox_weights, avg_factor=num_pos)
        loss_dict = dict(loss_cls=loss_cls, loss_iou=loss_iou, loss_obj=loss_obj)

        if self.use_l1:
            loss_box = self.loss_box(flatten_bbox_decode.view(-1, 4)[pos_masks], l1_targets, weight=bbox_weights, avg_factor=num_pos)
            loss_dict.update(loss_box=loss_box)

        if self.use_dfl:
            # 参考 gfl_head.py Line287-291
            # # forked from gfl_head.py  Line264  =>loss_iou反向升高！？？
            weight_targets = flatten_cls_preds.view(-1, self.cls_out_channels).detach().sigmoid()
            weight_targets = weight_targets.max(dim=1)[0][pos_masks]
            pred_corners = flatten_bbox_preds.view(-1, 4*self.reg_val['num'])[pos_masks].view(-1, self.reg_val['num'])
            priors_targets = torch.cat([flatten_mlvl_priors] * num_imgs, dim=0)[pos_masks]
            target_corners = self.distpoint_coder.encode(self.anchor_center(priors_targets), bbox_targets, self.reg_val['max']).reshape(-1)
            loss_dfl = self.loss_dfl(pred_corners, target_corners, weight=weight_targets[:, None].expand(-1, 4).reshape(-1), avg_factor=4.0)
            loss_dict.update({'loss_dfl': loss_dfl})

        if self.has_teacher and teacher_info:
            if self.loss_distill == 'v0':
                loss_distill = dict()
            elif self.loss_distill == 'v1':
                loss_distill = self.loss_distill_v1(
                    flatten_cls_preds, flatten_bbox_preds, flatten_mlvl_priors, flatten_bbox_decode,
                    teacher_info, student_feat, pos_masks, num_pos, num_imgs, imgs_whwh, img_metas)
            elif self.loss_distill == 'v2':
                loss_distill = self.loss_distill_v2(
                    flatten_cls_preds, flatten_bbox_preds, flatten_mlvl_priors, flatten_bbox_decode,
                    teacher_info, student_feat, pos_masks, num_pos, num_imgs, imgs_whwh, img_metas)
            else:
                raise NotImplementedError
            loss_distill = {k: v * self.alpha_distill for k, v in loss_distill.items()}
            loss_dict.update(loss_distill)

        return loss_dict

    def loss_distill_v2(self, batch_cls_score, batch_box_score, batch_mlvl_priors, batch_box_decode,
                        teacher_info, student_feat, pos_masks, num_pos, num_imgs, imgs_whwh, img_metas):
        loss_dict = {}
        if 'keepindex' in self.which_index:
            keepidx = teacher_info['pred_keepid']
        else:
            keepidx = pos_masks
        keep_num = keepidx.nonzero().numel()
        reg_num = self.reg_val['num']
        cd_T, ld_T = getattr(self.loss_cd_soft, 'T', None), getattr(self.loss_ld_soft, 'T', None)

        # if 'kldv' in self.feats_distill:
        #     # assert len(student_feat) == len(teacher_info['neck_feats'])
        #     print('sf != tf => ', [(sf != tf).nonzero().numel() for sf, tf in
        #                            zip(student_feat, teacher_info['neck_feats'])])
        #     print('sf == tf => ', [(sf == tf).nonzero().numel() for sf, tf in
        #                            zip(student_feat, teacher_info['neck_feats'])])
        #     loss_fd = [self.loss_fd(sf, tf, weight=None, avg_factor=num_imgs)
        #                for sf, tf in zip(student_feat, teacher_info['neck_feats'])]
        #     loss_dict.update({'loss_fd': loss_fd})

        # 获得输出
        imgs_whwh = imgs_whwh[keepidx]

        batch_cls_score = batch_cls_score.view(-1, self.cls_out_channels)[keepidx]
        soft_cls_score = [cls_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.cls_out_channels)
                          for cls_pred in teacher_info['head_outs'][0]]
        soft_cls_score = torch.cat(soft_cls_score, dim=1).view(-1, self.cls_out_channels)[keepidx]

        batch_box_score = batch_box_score.view(-1, self.reg_out_channels)[keepidx]
        soft_box_score = [box_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.reg_out_channels)
                           for box_pred in teacher_info['head_outs'][1]]
        soft_box_score = torch.cat(soft_box_score, dim=1).view(-1, self.reg_out_channels)[keepidx]

        # 输出激活
        batch_cls_score = batch_cls_score.sigmoid()
        soft_cls_score = soft_cls_score.sigmoid()
        if self.normal_clsscore or self.loss_cd_soft._get_name() == 'KnowledgeDistillationKLDivLoss':
            batch_cls_score = batch_cls_score / batch_cls_score.sum(-1, keepdim=True)
            soft_cls_score = soft_cls_score / batch_cls_score.sum(-1, keepdim=True)
        if self.use_dfl:
            batch_box_score = myactivate(batch_box_score, func=self.reg_val['activate'], dim=-1)
            soft_box_score = myactivate(soft_box_score, func=self.reg_val['activate'], dim=-1)

        # 输出解码
        if 'decode' in self.locat_distill:
            soft_box_decode = self._bbox_decode(batch_mlvl_priors, soft_box_score)
            # soft_box_decode = soft_box_decode.view(-1, 4)[keepidx]
            batch_box_decode = batch_box_decode.view(-1, 4)[keepidx]
        else:
            soft_box_decode = soft_box_score
            batch_box_decode = batch_box_decode.view(-1, 4)[keepidx]

        # 输出调节
        if self.hybrid_score and keep_num > 0:
            # x[F.one_hot(x.max(-1)[1], x.size(-1)).bool()].view(x.size()[:-1]) == x.max(-1)[0]
            soft_cls_score[F.one_hot(soft_cls_score.max(-1)[1], self.num_classes).bool()] = 1.0
            if self.use_dfl:
                soft_box_score[F.one_hot(soft_box_score.max(-1)[1], self.reg_out_channels).bool()] = 1.0

        # 计算差值
        if 'soft' in self.cates_distill:
            if self.loss_cd_soft._get_name() == 'KnowledgeDistillationKLDivLoss':
                cls_err = F.kl_div(
                    (batch_cls_score / cd_T).log(),
                    (soft_cls_score / cd_T).detach(),
                    reduction='none').mean(-1) * (cd_T * cd_T)
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
        if self.loss_ld_bbox and self.loss_ld_bbox._get_name() == 'SmoothL1Loss':
            l1dist = torch.where(l1dist < 1.0, 0.5 * l1dist * l1dist / 1.0, l1dist - 0.5 * 1.0)
        xgiou = 1 - bbox_overlaps(batch_box_decode, soft_box_decode, mode='giou', is_aligned=True, eps=1e-7)

        # 计算损失权重
        cls_weight, loc_weight, bbox_weight, iou_weight = [None] * 4
        if self.mixed_clsloc and keep_num > 0:
            # N阶协同多任务损失函数；+为0阶，相互包含N次为N阶；
            # 连续学习下的分类/定位协同样本挖掘：共同样本挖掘：0~2之间，1处代表分类定位的平衡，
            # (cls, loc) = (坏,坏)，(坏,好)，(好,坏)，(好,好).
            cls_weight = 2 * cls_err / (cls_err + loc_err)
            loc_weight = 2 * loc_err / (cls_err + loc_err)

        # 计算损失
        if 'kldv' in self.feats_distill:
            # assert len(student_feat) == len(teacher_info['neck_feats'])
            print('sf != tf => ', [(sf != tf).nonzero().numel() for sf, tf in
                                   zip(student_feat, teacher_info['neck_feats'])])
            print('sf == tf => ', [(sf == tf).nonzero().numel() for sf, tf in
                                   zip(student_feat, teacher_info['neck_feats'])])
            loss_fd = [self.loss_fd(sf, tf, weight=None, avg_factor=num_imgs)
                       for sf, tf in zip(student_feat, teacher_info['neck_feats'])]
            loss_dict.update({'loss_fd': loss_fd})

        if 'soft' in self.cates_distill:
            avg_factor = keep_num
            loss_cd_soft = self.loss_cd_soft.loss_weight * reduce_loss(
                           cls_err, cls_weight, self.loss_cd_soft.reduction, avg_factor)
            loss_cd_soft1 = self.loss_cd_soft(batch_cls_score, soft_cls_score, weight=cls_weight, avg_factor=avg_factor)
            # assert loss_cd_soft1 == loss_cd_soft
            loss_dict.update({'loss_cd_soft': loss_cd_soft})

        if 'soft' in self.locat_distill:
            avg_factor = keep_num * 1
            loss_ld_soft = self.loss_ld_soft.loss_weight * reduce_loss(
                           loc_err, loc_weight, self.loss_ld_soft.reduction, avg_factor)
            loss_ld_soft1 = self.loss_ld_soft(batch_box_score, soft_box_score, weight=loc_weight, avg_factor=avg_factor)
            # assert loss_ld_soft1 == loss_ld_soft
            loss_dict.update({'loss_ld_soft': loss_ld_soft})

        if 'box' in self.locat_distill:
            avg_factor = keep_num
            loss_ld_bbox = self.loss_ld_bbox.loss_weight * reduce_loss(
                           l1dist, bbox_weight, self.loss_ld_bbox.reduction, avg_factor)
            loss_ld_bbox1 = self.loss_ld_bbox(batch_box_decode / imgs_whwh, soft_box_decode / imgs_whwh, weight=bbox_weight, avg_factor=avg_factor)
            # assert loss_ld_bbox1 == loss_ld_bbox
            loss_dict.update({'loss_ld_bbox': loss_ld_bbox})

        if 'iou' in self.locat_distill:
            avg_factor = keep_num
            loss_ld_iou = self.loss_ld_iou.loss_weight * reduce_loss(
                          xgiou, iou_weight, self.loss_ld_iou.reduction, avg_factor)
            loss_ld_iou1 = self.loss_ld_iou(batch_box_decode, soft_box_decode, weight=iou_weight, avg_factor=avg_factor)
            # assert loss_ld_iou1 == loss_ld_iou
            loss_dict.update({'loss_ld_iou': loss_ld_iou})

        if keep_num == 0:
            assert all([x.item() == 0 for x in loss_dict.values()]), print(f'keep_num={keep_num}, {loss_dict}')

        return loss_dict

    def loss_distill_v1(self, batch_cls_score, batch_box_score, batch_mlvl_priors, batch_box_decode,
                        teacher_info, student_feat, pos_masks, num_pos, num_imgs, imgs_whwh, img_metas):
        loss_dict = {}
        if 'keepindex' in self.which_index:
            keepidx = teacher_info['pred_keepid']
        else:
            keepidx = pos_masks
        keep_num = keepidx.numel()

        if 'kldv' in self.feats_distill:
            # assert len(student_feat) == len(teacher_info['neck_feats'])
            # print('sf != tf => ', [(sf != tf).nonzero().numel() for sf, tf in
            #                        zip(student_feat, teacher_info['neck_feats'])])
            loss_fd = [self.loss_fd(sf, tf, weight=None, avg_factor=sf.numel())
                       for sf, tf in zip(student_feat, teacher_info['neck_feats'])]
            loss_dict.update({'loss_fd': loss_fd})

        if 'soft' in self.cates_distill and False:
            batch_cls_score = batch_cls_score.view(-1, self.cls_out_channels)[keepidx]
            soft_cls_score = [cls_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.cls_out_channels)
                              for cls_pred in teacher_info['head_outs'][0]]
            soft_cls_score = torch.cat(soft_cls_score, dim=1)
            soft_cls_score = soft_cls_score.view(-1, self.cls_out_channels)[keepidx]
            if self.hybrid_score:
                # x[F.one_hot(x.max(-1)[1], x.size(-1)).bool()].view(x.size()[:-1]) == x.max(-1)[0]
                soft_cls_score[F.one_hot(soft_cls_score.max(-1)[1], self.num_classes).bool()] = 1.0
            loss_cd_soft = self.loss_cd_soft(batch_cls_score, soft_cls_score,
                                             weight=None, avg_factor=keep_num)
            soft_cls_score = None
            loss_dict.update({'loss_cd_soft': loss_cd_soft})

        if 'soft' in self.cates_distill:
            batch_cls_score = batch_cls_score.view(-1, self.cls_out_channels)[keepidx]
            soft_cls_score = [cls_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.cls_out_channels)
                              for cls_pred in teacher_info['head_outs'][0]]
            soft_cls_score = torch.cat(soft_cls_score, dim=1).view(-1, self.cls_out_channels)[keepidx]

            batch_cls_score = batch_cls_score.sigmoid()
            soft_cls_score = soft_cls_score.sigmoid()

            if self.normal_clsscore:    # 只在KLDV损失下使用
                batch_cls_score = batch_cls_score / batch_cls_score.sum(-1, keepdim=True)
                soft_cls_score = soft_cls_score / batch_cls_score.sum(-1, keepdim=True)
            if self.hybrid_score:
                # x[F.one_hot(x.max(-1)[1], x.size(-1)).bool()].view(x.size()[:-1]) == x.max(-1)[0]
                soft_cls_score[F.one_hot(soft_cls_score.max(-1)[1], self.num_classes).bool()] = 1.0

            cls_err = None
            if self.loss_cd_soft._get_name() == 'KnowledgeDistillationKLDivLoss':
                cd_T = self.loss_cd_soft.T
                cls_err = F.kl_div(
                    (batch_cls_score / cd_T).log(),
                    (soft_cls_score / cd_T).detach(),
                    reduction='none').mean(-1) * (cd_T * cd_T)
            elif self.loss_cd_soft._get_name() == 'FocalLoss':
                # forked from py_focal_loss_with_prob()
                pred = batch_cls_score
                target = soft_cls_score.type_as(pred)
                alpha, gamma = self.loss_cd_soft.alpha, self.loss_cd_soft.gamma
                pt = (1 - pred) * target + pred * (1 - target)
                focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * pt.pow(gamma)
                cls_err = F.binary_cross_entropy(pred, target, reduction='none') * focal_weight
                cls_err = cls_err.sum(-1)       # 对(keep_num, num_classes)求和
            elif self.loss_cd_soft._get_name() == 'MSELoss':
                cls_err = F.mse_loss(
                    batch_cls_score,
                    soft_cls_score,
                    reduction='none').sum(-1)
            avg_factor = keep_num
            loss_cd_soft = self.loss_cd_soft.loss_weight * reduce_loss(
                cls_err, None, self.loss_cd_soft.reduction, avg_factor)
            loss_dict.update({'loss_cd_soft': loss_cd_soft})

        if 'soft' in self.locat_distill:
            # forked from ld_head.py Line99-Line122
            soft_box_score = [box_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.reg_out_channels)
                              for box_pred in teacher_info['head_outs'][1]]
            soft_box_score = torch.cat(soft_box_score, dim=1)
            batch_box_score = batch_box_score.view(-1, self.reg_out_channels)[keepidx]
            soft_box_score = soft_box_score.view(-1, self.reg_out_channels)[keepidx]
            loss_ld_soft = self.loss_ld_soft(batch_box_score, soft_box_score,
                                             weight=None,  avg_factor=keep_num)
            soft_box_score = None
            loss_dict.update({'loss_ld_soft': loss_ld_soft})

        if 'box' in self.locat_distill:
            soft_box_score = [box_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.reg_out_channels)
                              for box_pred in teacher_info['head_outs'][1]]
            soft_box_score = torch.cat(soft_box_score, dim=1)
            if 'decode' in self.locat_distill:
                soft_box_score = self._bbox_decode(batch_mlvl_priors, soft_box_score)
                batch_pred_bbox = batch_box_decode
            else:
                batch_pred_bbox = batch_box_score
            batch_pred_bbox = batch_pred_bbox.view(-1, 4)[keepidx]
            soft_box_score = soft_box_score.view(-1, 4)[keepidx]
            loss_ld_bbox = self.loss_ld_bbox(batch_pred_bbox, soft_box_score,
                                             weight=None, avg_factor=keep_num)
            soft_box_score = None
            loss_dict.update({'loss_ld_bbox': loss_ld_bbox})

        if 'iou' in self.locat_distill:
            soft_box_score = [box_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.reg_out_channels)
                              for box_pred in teacher_info['head_outs'][1]]
            soft_box_score = torch.cat(soft_box_score, dim=1)
            if 'decode' in self.locat_distill:
                soft_box_score = self._bbox_decode(batch_mlvl_priors, soft_box_score)
                batch_pred_bbox = batch_box_decode
            else:
                batch_pred_bbox = batch_box_score
            loss_ld_iou = self.loss_ld_iou(batch_pred_bbox, soft_box_score,
                                           weight=None, avg_factor=keepidx.numel())
            soft_box_score = None
            loss_dict.update({'loss_ld_iou': loss_ld_iou})

        return loss_dict

    @torch.no_grad()
    def _get_target_single(self, cls_preds, objectness, priors, decoded_bboxes, gt_bboxes, gt_labels, img_meta):
        """Compute classification, regression, and objectness targets for
        priors in a single image.
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
            return (foreground_mask, cls_target, obj_target, bbox_target, l1_target, 0)

        # YOLOY uses center priors with 0.5 offset to assign targets,
        # but use center priors without offset to regress bboxes.
        offset_priors = torch.cat([priors[:, :2] + priors[:, 2:] * 0.5, priors[:, 2:]], dim=-1)

        assign_result = self.assigner.assign(
            cls_preds.sigmoid() * objectness.unsqueeze(1).sigmoid(),
            offset_priors, decoded_bboxes, gt_bboxes, gt_labels)

        sampling_result = self.sampler.sample(assign_result, priors, gt_bboxes)
        pos_inds = sampling_result.pos_inds
        num_pos_per_img = pos_inds.size(0)

        h, w, _ = img_meta['img_shape']
        img_whwh = gt_bboxes.new_tensor([[w, h, w, h]])
        img_whwh = img_whwh.repeat(num_pos_per_img, 1)

        pos_ious = assign_result.max_overlaps[pos_inds]
        # IOU aware classification score
        cls_target = F.one_hot(sampling_result.pos_gt_labels, self.num_classes) * pos_ious.unsqueeze(-1)
        obj_target = torch.zeros_like(objectness).unsqueeze(-1)
        obj_target[pos_inds] = 1
        bbox_target = sampling_result.pos_gt_bboxes
        l1_target = cls_preds.new_zeros((num_pos_per_img, 4))
        if self.use_l1:
            l1_target = self._get_l1_target(l1_target, bbox_target, priors[pos_inds])
        foreground_mask = torch.zeros_like(objectness).to(torch.bool)
        foreground_mask[pos_inds] = 1
        return (foreground_mask, cls_target, obj_target, bbox_target, l1_target, img_whwh, num_pos_per_img)

    def _get_l1_target(self, l1_target, gt_bboxes, priors, eps=1e-8):
        """Convert gt bboxes to center offset and log width height."""
        gt_cxcywh = bbox_xyxy_to_cxcywh(gt_bboxes)
        l1_target[:, :2] = (gt_cxcywh[:, :2] - priors[:, :2]) / priors[:, 2:]
        l1_target[:, 2:] = torch.log(gt_cxcywh[:, 2:] / priors[:, 2:] + eps)
        return l1_target

    def Xsimple_test(self, feats, img_metas, rescale=False):
        super(YOLOYHead, self).simple_test(feats, img_metas, rescale)

    def Xsimple_test_bboxes(self, feats, img_metas, rescale=False):
        """Test det bboxes without test-time augmentation, can be applied in
        DenseHead except for ``RPNHead`` and its variants, e.g., ``GARPNHead``,
        etc.
        """
        # super(YOLOYHead, self).simple_test_bboxes(feats, img_metas, rescale=False)
        outs = self.forward(feats)
        results_list = self.get_bboxes(*outs, img_metas=img_metas, rescale=rescale)
        return results_list

    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   objectnesses,
                   img_metas=None,
                   cfg=None,
                   rescale=False,
                   with_nms=True,
                   need_keepid=False):
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

        flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_objectness = torch.cat(flatten_objectness, dim=1).sigmoid()
        flatten_mlvl_priors = torch.cat(mlvl_priors)
        flatten_bbox_decode = self._bbox_decode(flatten_mlvl_priors, flatten_bbox_preds)
        if rescale:
            flatten_bbox_decode[..., :4] /= flatten_bbox_decode.new_tensor(scale_factors).unsqueeze(1)

        result_list = []
        for img_id in range(len(img_metas)):
            cls_scores = flatten_cls_scores[img_id]
            obj_scores = flatten_objectness[img_id]
            box_decode = flatten_bbox_decode[img_id]
            result = self._bboxes_nms(cls_scores, box_decode, obj_scores, cfg, need_keepid)
            result_list.append(result)

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

    def _bboxes_nms(self, cls_scores, bboxes, score_factor, cfg, need_keepid=False):
        max_scores, labels = torch.max(cls_scores, 1)
        valid_mask = score_factor * max_scores >= cfg.score_thr

        bboxes = bboxes[valid_mask]
        scores = max_scores[valid_mask] * score_factor[valid_mask]
        labels = labels[valid_mask]

        if labels.numel() == 0:
            dets, labels, valid_mask = bboxes, labels, valid_mask
        else:
            dets, keep = batched_nms(bboxes, scores, labels, cfg.nms)
            labels = labels[keep]
            # valid_mask = valid_mask.nonzero()[keep].flatten()
            valid_mask = valid_mask.nonzero(as_tuple=True)[0][keep]
        if need_keepid:
            return dets, labels, valid_mask
        else:
            return dets, labels