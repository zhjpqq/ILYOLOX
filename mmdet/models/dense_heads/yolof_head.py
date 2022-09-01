# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (ConvModule, bias_init_with_prob, constant_init, is_norm, normal_init)
from mmcv.runner import force_fp32

from mmdet.core import anchor_inside_flags, multi_apply, reduce_mean, unmap, build_bbox_coder
from ..builder import HEADS, build_loss
from .anchor_head import AnchorHead
from .anchor_head_il import AnchorHeadIL
from mmdet.core.utils import filter_scores_and_topk, select_single_mlvl
from mmcv.ops import batched_nms
from mmdet.core.bbox.coder import DistancePointBBoxCoder
INF = 1e8
from ..utils.misc import Integral


def levels_to_images(mlvl_tensor):
    """Concat multi-level feature maps by image.

    [feature_level0, feature_level1...] -> [feature_image0, feature_image1...]
    Convert the shape of each element in mlvl_tensor from (N, C, H, W) to
    (N, H*W , C), then split the element to N elements with shape (H*W, C), and
    concat elements in same image of all level along first dimension.

    Args:
        mlvl_tensor (list[torch.Tensor]): list of Tensor which collect from
            corresponding level. Each element is of shape (N, C, H, W)

    Returns:
        list[torch.Tensor]: A list that contains N tensors and each tensor is
            of shape (num_elements, C)
    """
    batch_size = mlvl_tensor[0].size(0)
    batch_list = [[] for _ in range(batch_size)]
    channels = mlvl_tensor[0].size(1)
    for t in mlvl_tensor:
        t = t.permute(0, 2, 3, 1)
        t = t.view(batch_size, -1, channels).contiguous()
        for img in range(batch_size):
            batch_list[img].append(t[img])
    return [torch.cat(item, 0) for item in batch_list]


@HEADS.register_module()
class YOLOFHead(AnchorHead):
    """YOLOFHead Paper link: https://arxiv.org/abs/2103.09460.

    Args:
        num_classes (int): The number of object classes (w/o background)
        in_channels (List[int]): The number of input channels per scale.
        cls_num_convs (int): The number of convolutions of cls branch.
           Default 2.
        reg_num_convs (int): The number of convolutions of reg branch.
           Default 4.
        norm_cfg (dict): Dictionary to construct and config norm layer.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 num_cls_convs=2,
                 num_reg_convs=4,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 cates_distill='',
                 locat_distill='',
                 feats_distill='',
                 reg_val={'min':0, 'max':16, 'num':17, 'usedfl': False},
                 loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25),
                 loss_kd=dict(
                     type='KnowledgeDistillationKLDivLoss',
                     loss_weight=10,
                     T=2),
                 loss_ld_bbox=dict(type='SmoothL1Loss', loss_weight=10, reduction='mean'),
                 loss_ld_logit=dict(
                     type='KnowledgeDistillationKLDivLoss',
                     loss_weight=0.25,
                     T=10),
                 loss_fd=dict(
                     type='KnowledgeDistillationKLDivLoss',
                     loss_weight=10,
                     T=2),
                 **kwargs):
        self.num_cls_convs = num_cls_convs
        self.num_reg_convs = num_reg_convs
        self.norm_cfg = norm_cfg
        self.reg_val = reg_val
        self.use_dfl = self.reg_val['usedfl']
        self.has_teacher = kwargs.pop('has_teacher', False)
        super(YOLOFHead, self).__init__(num_classes, in_channels, **kwargs)

        if self.use_dfl:
            self.integral = Integral(reg_val)
            self.loss_dfl = build_loss(loss_dfl)
            self.distpoint_coder = build_bbox_coder(dict(type='DistancePointBBoxCoder', clip_border=True))

        self.cates_distill = cates_distill
        self.locat_distill = locat_distill
        self.feats_distill = feats_distill
        self.loss_kd = build_loss(loss_kd) if cates_distill else None
        self.loss_ld_bbox = build_loss(loss_ld_bbox) if 'bbox' in locat_distill else None
        self.loss_ld_logit = build_loss(loss_ld_logit) if 'logit' in locat_distill and self.use_dfl else None
        self.loss_fd = build_loss(loss_fd) if feats_distill else None

    def _init_layers(self):
        # used in super class
        cls_subnet = []
        bbox_subnet = []
        for i in range(self.num_cls_convs):
            cls_subnet.append(
                ConvModule(
                    self.in_channels,
                    self.in_channels,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=self.norm_cfg))
        for i in range(self.num_reg_convs):
            bbox_subnet.append(
                ConvModule(
                    self.in_channels,
                    self.in_channels,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=self.norm_cfg))
        self.cls_subnet = nn.Sequential(*cls_subnet)
        self.bbox_subnet = nn.Sequential(*bbox_subnet)
        self.cls_score = nn.Conv2d(
            self.in_channels,
            self.num_base_priors * self.num_classes,
            kernel_size=3,
            stride=1,
            padding=1)
        self.bbox_pred = nn.Conv2d(
            self.in_channels,
            self.num_base_priors * 4 * (self.reg_val['num'] if self.use_dfl else 1),
            kernel_size=3,
            stride=1,
            padding=1)
        self.object_pred = nn.Conv2d(
            self.in_channels,
            self.num_base_priors,
            kernel_size=3,
            stride=1,
            padding=1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, mean=0, std=0.01)
            if is_norm(m):
                constant_init(m, 1)

        # Use prior in model initialization to improve stability
        bias_cls = bias_init_with_prob(0.01)
        torch.nn.init.constant_(self.cls_score.bias, bias_cls)

    def forward(self, feats):
        """Forward features from the upstream network.
        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
        Returns:
            tuple: A tuple of classification scores and bbox prediction.
                - cls_scores (list[Tensor]): Classification scores for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_base_priors * num_classes.
                - bbox_preds (list[Tensor]): Box energies / deltas for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_base_priors * 4.
        """
        return multi_apply(self.forward_single, feats)

    def forward_single(self, feature):
        cls_score = self.cls_score(self.cls_subnet(feature))
        N, _, H, W = cls_score.shape
        cls_score = cls_score.view(N, -1, self.num_classes, H, W)

        reg_feat = self.bbox_subnet(feature)
        bbox_reg = self.bbox_pred(reg_feat)
        objectness = self.object_pred(reg_feat)

        # implicit objectness
        objectness = objectness.view(N, -1, 1, H, W)
        normalized_cls_score = cls_score + objectness - torch.log(
            1. + torch.clamp(cls_score.exp(), max=INF) +
            torch.clamp(objectness.exp(), max=INF))
        normalized_cls_score = normalized_cls_score.view(N, -1, H, W)
        return normalized_cls_score, bbox_reg

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (list[Tensor]): Proposals of each image.
        """
        teacher_info = kwargs.pop('teacher_info', {})
        student_feat = x if self.has_teacher and self.feats_distill else []

        outs = self.forward(x)
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore,
                           student_feat=student_feat, teacher_info=teacher_info)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(*outs, img_metas=img_metas, cfg=proposal_cfg)
            return losses, proposal_list

    def simple_test(self, feats, img_metas, rescale=False):
        """Test function without test-time augmentation.

        Args:
            feats (tuple[torch.Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is ``bboxes`` with shape (n, 5),
                where 5 represent (tl_x, tl_y, br_x, br_y, score).
                The shape of the second tensor in the tuple is ``labels``
                with shape (n, ).
        """
        return self.simple_test_bboxes(feats, img_metas, rescale=rescale)

    def simple_test_bboxes(self, feats, img_metas, rescale=False):
        """Test det bboxes without test-time augmentation, can be applied in
        DenseHead except for ``RPNHead`` and its variants, e.g., ``GARPNHead``,
        etc.
        Args:
            feats (tuple[torch.Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.
        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is ``bboxes`` with shape (n, 5),
                where 5 represent (tl_x, tl_y, br_x, br_y, score).
                The shape of the second tensor in the tuple is ``labels``
                with shape (n,)
        """
        outs = self.forward(feats)
        results_list = self.get_bboxes(*outs, img_metas=img_metas, rescale=rescale)
        return results_list

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None,
             student_feat=[],
             teacher_info={}):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (batch, num_anchors * num_classes, h, w)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (batch, num_anchors * 4, h, w)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss. Default: None

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert len(cls_scores) == 1
        assert self.prior_generator.num_levels == 1

        device = cls_scores[0].device
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        anchor_list, valid_flag_list = self.get_anchors(featmap_sizes, img_metas, device=device)

        # The output level is always 1
        anchor_list = [anchors[0] for anchors in anchor_list]
        valid_flag_list = [valid_flags[0] for valid_flags in valid_flag_list]
        # [(HW, PC), ...]  [(HW, P4), ...]
        cls_scores_list = levels_to_images(cls_scores)
        bbox_preds_list = levels_to_images(bbox_preds)

        # 合并 GT-Label-Boxes & Teacher-Label-Boxes
        if self.has_teacher and 'hard' in self.cates_distill:
            for i in range(len(img_metas)):
                gt_labels[i] = torch.cat([teacher_info['pred_labels'][i], gt_labels[i]], dim=0)
                gt_bboxes[i] = torch.cat([teacher_info['pred_bboxes'][i], gt_bboxes[i]], dim=0)

        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(
            cls_scores_list,
            bbox_preds_list,
            anchor_list,
            valid_flag_list,
            gt_bboxes_list=gt_bboxes,
            img_metas=img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None
        (batch_labels, batch_label_weights, num_total_pos, num_total_neg,
         batch_bbox_weights, batch_pospred_boxes, batch_target_boxes) = cls_reg_targets

        # [(i, x.item()) for i, x in enumerate(flatten_labels) if x in gt_labels]
        # len([(i, x.item()) for i, x in enumerate(flatten_label_weights) if x == 1]) == num_total_pos + num_total_neg
        # cls_score: [B, PC, H, W] => [BHWP, C] => batch_cls_score
        batch_cls_score = cls_scores[0].permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)

        num_total_samples = (num_total_pos + num_total_neg) if self.sampling else num_total_pos
        num_total_samples = reduce_mean(batch_cls_score.new_tensor(num_total_samples)).clamp_(1.0).item()

        # TODO IL，针对新旧任务类别设置不同的权重系数
        # flatten_label_weights[flatten_labels in prev_task_gt_labels] *= 0.8

        # TODO IL，针对新旧任务盒子设置不同的权重系数
        # batch_bbox_weights

        # classification loss
        loss_cls = self.loss_cls(
            batch_cls_score,
            batch_labels,
            batch_label_weights,
            avg_factor=num_total_samples)

        # regression loss
        if batch_pospred_boxes.shape[0] == 0:
            # no pos sample
            loss_bbox = batch_pospred_boxes.sum() * 0
        else:
            loss_bbox = self.loss_bbox(
                batch_pospred_boxes,
                batch_target_boxes,
                batch_bbox_weights.float(),
                avg_factor=num_total_samples)

        loss_dict = dict(loss_cls=loss_cls, loss_bbox=loss_bbox)

        if self.use_dfl and False:
            # gfl_head.py Line287-291
            pred_corners = batch_pospred_boxes.reshape(-1, self.reg_val['num'])
            target_corners = self.distpoint_coder.encode(pos_anchor_centers, batch_target_boxes, self.reg_val['max']).reshape(-1)
            loss_dfl = self.loss_dfl(
                pred_corners,
                target_corners,
                weight=weight_targets[:, None].expand(-1, 4).reshape(-1),
                avg_factor=4.0)
            loss_dict.update({'loss_dfl': loss_dfl})

        if self.has_teacher:
            # TODO 背景类别在pred_keepid中被去除 ？ 只有高置信度正样本做Loss！
            if 'soft' in self.cates_distill:
                soft_label = teacher_info['head_outs'][0][0].permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
                soft_weight = soft_label.new_zeros(size=(soft_label.shape[0], 1))
                soft_weight[teacher_info['pred_keepid']] = 1
                loss_kd = self.loss_kd(batch_cls_score, soft_label, weight=soft_weight,
                                       avg_factor=soft_weight[soft_weight==1].numel())
                loss_dict.update({'loss_kd': loss_kd})
            if 'bbox' in self.locat_distill:
                batch_pred_bbox = bbox_preds[0].permute(0, 2, 3, 1)#.reshape(-1, 4*self.reg_val['num'])
                batch_soft_bbox = teacher_info['head_outs'][1][0].permute(0, 2, 3, 1)#.reshape(-1, 4*self.reg_val['num'])
                batch_pred_bbox = self.integral(batch_pred_bbox, reshape=True)
                batch_soft_bbox = self.integral(batch_soft_bbox, reshape=True)
                soft_weight = batch_soft_bbox.new_zeros(size=(batch_soft_bbox.shape[0], 1))
                soft_weight[teacher_info['pred_keepid']] = 1
                loss_ld_bbox = self.loss_ld_bbox(batch_pred_bbox, batch_soft_bbox, weight=soft_weight,
                                                 avg_factor=soft_weight[soft_weight==1].numel())
                loss_dict.update({'loss_ld_bbox': loss_ld_bbox})
            if 'logit' in self.locat_distill and self.use_dfl:
                # forked from ld_head.py Line99-Line122
                batch_pred_bbox = bbox_preds[0].permute(0, 2, 3, 1).reshape(-1, 4*self.reg_val['num'])
                batch_soft_bbox = teacher_info['head_outs'][1][0].permute(0, 2, 3, 1).reshape(-1, 4*self.reg_val['num'])
                soft_weight = batch_soft_bbox.new_zeros(size=(batch_soft_bbox.shape[0], 1))
                soft_weight[teacher_info['pred_keepid']] = 1
                loss_ld_logit = self.loss_ld_logit(batch_pred_bbox, batch_soft_bbox, weight=soft_weight,
                                                   avg_factor=soft_weight[soft_weight==1].numel())
                loss_dict.update({'loss_ld_logit': loss_ld_logit})
            if 'kldv' in self.feats_distill:
                # assert len(student_feat) == len(teacher_info['neck_feats'])
                loss_fd = [self.loss_fd(sf, tf, weight=None, avg_factor=None)
                           for sf, tf in zip(student_feat, teacher_info['neck_feats'])]
                avg_factor = [1, len(loss_fd), len(img_metas), num_total_samples][2]
                loss_fd = sum(loss_fd)/avg_factor
                loss_dict.update({'loss_fd': loss_fd})
        return loss_dict

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    anchor_list,
                    valid_flag_list,
                    gt_bboxes_list,
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    label_channels=1,
                    unmap_outputs=True):
        """Compute regression and classification targets for anchors in
        multiple images.

        Args:
            cls_scores_list (list[Tensor])： Classification scores of
                each image. each is a 4D-tensor, the shape is
                (h * w, num_anchors * num_classes).
            bbox_preds_list (list[Tensor])： Bbox preds of each image.
                each is a 4D-tensor, the shape is (h * w, num_anchors * 4).
            anchor_list (list[Tensor]): Anchors of each image. Each element of
                is a tensor of shape (h * w * num_anchors, 4).
            valid_flag_list (list[Tensor]): Valid flags of each image. Each
               element of is a tensor of shape (h * w * num_anchors, )
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
            img_metas (list[dict]): Meta info of each image.
            gt_bboxes_ignore_list (list[Tensor]): Ground truth bboxes to be
                ignored.
            gt_labels_list (list[Tensor]): Ground truth labels of each box.
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple: Usually returns a tuple containing learning targets.

                - batch_labels (Tensor): Label of all images. Each element \
                    of is a tensor of shape (batch, h * w * num_anchors)
                - batch_label_weights (Tensor): Label weights of all images \
                    of is a tensor of shape (batch, h * w * num_anchors)
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
            additional_returns: This function enables user-defined returns from
                `self._get_targets_single`. These returns are currently refined
                to properties at each feature map (i.e. having HxW dimension).
                The results will be concatenated after the end
        """
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        results = multi_apply(
            self._get_targets_single,
            bbox_preds_list,
            anchor_list,
            valid_flag_list,
            gt_bboxes_list,
            gt_bboxes_ignore_list,
            gt_labels_list,
            img_metas,
            label_channels=label_channels,
            unmap_outputs=unmap_outputs)
        (all_labels, all_label_weights, pos_inds_list, neg_inds_list, sampling_results_list) = results[:5]
        rest_results = list(results[5:])        # user-added return values
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])

        batch_labels = torch.stack(all_labels, 0).reshape(-1)
        batch_label_weights = torch.stack(all_label_weights, 0).reshape(-1)

        res = (batch_labels, batch_label_weights, num_total_pos, num_total_neg)
        for i, rests in enumerate(rest_results):  # user-added return values
            rest_results[i] = torch.cat(rests, 0)

        return res + tuple(rest_results)

    def _get_targets_single(self,
                            bbox_preds,
                            flat_anchors,
                            valid_flags,
                            gt_bboxes,
                            gt_bboxes_ignore,
                            gt_labels,
                            img_meta,
                            label_channels=1,
                            unmap_outputs=True):
        """Compute regression and classification targets for anchors in a
        single image.

        Args:
            bbox_preds (Tensor): Bbox prediction of the image, which
                shape is (h * w ,num_anchors * 4)
            flat_anchors (Tensor): Anchors of the image, which shape is
                (h * w * num_anchors ,4)
            valid_flags (Tensor): Valid flags of the image, which shape is
                (h * w * num_anchors,).
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            img_meta (dict): Meta info of the image.
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple:
                labels (Tensor): Labels of image, which shape is
                    (h * w * num_anchors, ).
                label_weights (Tensor): Label weights of image, which shape is
                    (h * w * num_anchors, ).
                pos_inds (Tensor): Pos index of image.
                neg_inds (Tensor): Neg index of image.
                sampling_result (obj:`SamplingResult`): Sampling result.
                pos_bbox_weights (Tensor): The Weight of using to calculate
                    the bbox branch loss, which shape is (num, ).
                pos_predicted_boxes (Tensor): boxes predicted value of
                    using to calculate the bbox branch loss, which shape is
                    (num, 4).
                pos_target_boxes (Tensor): boxes target value of
                    using to calculate the bbox branch loss, which shape is
                    (num, 4).
        """
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None, ) * 8
        # assign gt and sample anchors
        if not self.use_dfl:
            bbox_preds = bbox_preds.reshape(-1, 4)
        else:
            bbox_preds = self.integral(bbox_preds, reshape=True)   # tblr分布=>tblr值
        bbox_preds = bbox_preds[inside_flags, :]
        anchors = flat_anchors[inside_flags, :]

        # decoded bbox
        # [(i, x.item()) for i, x in enumerate(assign_result.gt_inds) if x==1 ], [(1406, 1), (1411, 1), (1481, 1), (1486, 1)]
        # [(i, x.item()) for i, x in enumerate(assign_result.labels) if x!=-1 ], [(1406, 2), (1411, 2), (1481, 2), (1486, 2)]
        if not self.use_dfl:
            decoder_bbox_preds = self.bbox_coder.decode(anchors, bbox_preds)    # org dxdydwdh值=>x1y1x2y2值
        else:
            decoder_bbox_preds = self.distpoint_coder.decode(self.anchor_center(anchors), bbox_preds)  # tblr值=>x1y1x2y2值
        assign_result = self.assigner.assign(
            decoder_bbox_preds, anchors, gt_bboxes, gt_bboxes_ignore,
            None if self.sampling else gt_labels)

        pos_bbox_weights = assign_result.get_extra_property('pos_idx')      # ？？True & False
        pos_predicted_boxes = assign_result.get_extra_property('pos_predicted_boxes')
        pos_target_boxes = assign_result.get_extra_property('target_boxes')

        sampling_result = self.sampler.sample(assign_result, anchors, gt_bboxes)
        num_valid_anchors = anchors.shape[0]
        labels = anchors.new_full((num_valid_anchors, ), self.num_classes, dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            # TODO  gfl_head.py  _get_target_single.py
            # 传出位置分布预测，用在self.loss_dfl中！pos_bbox_weights != pos_inds ??
            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class since v2.5.0
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            labels = unmap(labels, num_total_anchors, inside_flags,
                           fill=self.num_classes)  # fill bg label  # TODO IL 背景类填充为0，方便增量学习
            label_weights = unmap(label_weights, num_total_anchors, inside_flags)

        return (labels, label_weights, pos_inds, neg_inds, sampling_result,
                pos_bbox_weights, pos_predicted_boxes, pos_target_boxes)

    def get_anchors(self, featmap_sizes, img_metas, device='cuda'):
        """Get anchors according to feature map sizes.
        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.
            device (torch.device | str): Device for returned tensors
        Returns:
            tuple:
                anchor_list (list[Tensor]): Anchors of each image.
                valid_flag_list (list[Tensor]): Valid flags of each image.
        """
        num_imgs = len(img_metas)

        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        multi_level_anchors = self.prior_generator.grid_priors(featmap_sizes, device=device)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]

        # for each image, we compute valid flags of multi level anchors
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = self.prior_generator.valid_flags(
                featmap_sizes, img_meta['pad_shape'], device)
            valid_flag_list.append(multi_level_flags)

        return anchor_list, valid_flag_list


    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   score_factors=None,
                   img_metas=None,
                   cfg=None,
                   rescale=False,
                   with_nms=True,
                   **kwargs):
        """Transform network outputs of a batch into bbox results.

        Note: When score_factors is not None, the cls_scores are
        usually multiplied by it then obtain the real score used in NMS,
        such as CenterNess in FCOS, IoU branch in ATSS.

        Args:
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * 4, H, W).
            score_factors (list[Tensor], Optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, num_priors * 1, H, W). Default None.
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
        assert len(cls_scores) == len(bbox_preds)

        if score_factors is None:
            # e.g. Retina, FreeAnchor, Foveabox, etc.
            with_score_factors = False
        else:
            # e.g. FCOS, PAA, ATSS, AutoAssign, etc.
            with_score_factors = True
            assert len(cls_scores) == len(score_factors)

        num_levels = len(cls_scores)

        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device)

        result_list = []

        for img_id in range(len(img_metas)):
            img_meta = img_metas[img_id]
            cls_score_list = select_single_mlvl(cls_scores, img_id)
            bbox_pred_list = select_single_mlvl(bbox_preds, img_id)
            if with_score_factors:
                score_factor_list = select_single_mlvl(score_factors, img_id)
            else:
                score_factor_list = [None for _ in range(num_levels)]

            # PreNMS|TopK|Threshold-on-EachLevel ==> PostNMS-on-AllLevel
            results = self._get_bboxes_single(cls_score_list, bbox_pred_list,
                                              score_factor_list, mlvl_priors,
                                              img_meta, cfg, rescale, with_nms,
                                              **kwargs)
            result_list.append(results)
        return result_list

    def _get_bboxes_single(self,
                           cls_score_list,
                           bbox_pred_list,
                           score_factor_list,
                           mlvl_priors,
                           img_meta,
                           cfg,
                           rescale=False,
                           with_nms=True,
                           **kwargs):
        """Transform outputs of a single image into bbox predictions.

        Args:
            cls_score_list (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_priors * num_classes, H, W).
            bbox_pred_list (list[Tensor]): Box energies / deltas from
                all scale levels of a single image, each item has shape
                (num_priors * 4, H, W).
            score_factor_list (list[Tensor]): Score factor from all scale
                levels of a single image, each item has shape
                (num_priors * 1, H, W).
            mlvl_priors (list[Tensor]): Each element in the list is
                the priors of a single level in feature pyramid. In all
                anchor-based methods, it has shape (num_priors, 4). In
                all anchor-free methods, it has shape (num_priors, 2)
                when `with_stride=True`, otherwise it still has shape
                (num_priors, 4).
            img_meta (dict): Image meta info.
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            tuple[Tensor]: Results of detected bboxes and labels. If with_nms
                is False and mlvl_score_factor is None, return mlvl_bboxes and
                mlvl_scores, else return mlvl_bboxes, mlvl_scores and
                mlvl_score_factor. Usually with_nms is False is used for aug
                test. If with_nms is True, then return the following format

                - det_bboxes (Tensor): Predicted bboxes with shape \
                    [num_bboxes, 5], where the first 4 columns are bounding \
                    box positions (tl_x, tl_y, br_x, br_y) and the 5-th \
                    column are scores between 0 and 1.
                - det_labels (Tensor): Predicted labels of the corresponding \
                    box with shape [num_bboxes].
                - det_logits (Tensor): Predicted logits of the corresponding \
                    box with shape [num_bboxes, num_classes].
        """
        if score_factor_list[0] is None:
            # e.g. Retina, FreeAnchor, etc.
            with_score_factors = False
        else:
            # e.g. FCOS, PAA, ATSS, etc.
            with_score_factors = True

        cfg = self.test_cfg if cfg is None else cfg
        img_shape = img_meta['img_shape']
        nms_pre = cfg.get('nms_pre', -1)

        mlvl_keepid = []
        mlvl_logits = []
        mlvl_scores = []
        mlvl_labels = []
        mlvl_bboxes = []
        if with_score_factors:
            mlvl_score_factors = []
        else:
            mlvl_score_factors = None
        for level_idx, (cls_score, bbox_pred, score_factor, priors) in \
                enumerate(zip(cls_score_list, bbox_pred_list, score_factor_list, mlvl_priors)):

            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

            if not self.use_dfl:
                bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)     # org
            else:
                bbox_pred = self.integral(bbox_pred.permute(1, 2, 0), reshape=True)
            if with_score_factors:
                score_factor = score_factor.permute(1, 2, 0).reshape(-1).sigmoid()
            cls_score = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                # remind that we set FG labels to [0, num_class-1]
                # since mmdet v2.0
                # BG cat_id: num_class
                scores = cls_score.softmax(-1)[:, :-1]

            # ① PreNMS 预先对每一个Level进行独立NMS，预先过滤
            # After https://github.com/open-mmlab/mmdetection/pull/6268/,
            # this operation keeps fewer bboxes under the same `nms_pre`.
            # There is no difference in performance for most models. If you
            # find a slight drop in performance, you can set a larger
            # `nms_pre` than before.
            results = filter_scores_and_topk(
                scores, cfg.score_thr, nms_pre,
                dict(bbox_pred=bbox_pred, priors=priors))
            scores, labels, keep_idxs, bbox_pred_prior = results
            # keep_idxs_list = keep_idxs.tolist()
            # print('keep_idxs1=>', len(keep_idxs_list)==len(set(keep_idxs_list)), len(keep_idxs_list), len(set(keep_idxs_list)))
            # print(keep_idxs_list)

            bbox_pred = bbox_pred_prior['bbox_pred']
            priors = bbox_pred_prior['priors']
            if not self.use_dfl:
                bboxes = self.bbox_coder.decode(priors, bbox_pred, max_shape=img_shape)    # org dxdydwdh值=>x1y1x2y2值
            else:
                bboxes = self.distpoint_coder.decode(self.anchor_center(priors), bbox_pred)  # tblr值=>x1y1x2y2值

            if with_score_factors:
                score_factor = score_factor[keep_idxs]

            mlvl_scores.append(scores)  # Nx1
            mlvl_labels.append(labels)  # Nx1
            mlvl_bboxes.append(bboxes)  # Nx4
            if with_score_factors:
                mlvl_score_factors.append(score_factor)
            if kwargs.get('need_logits', False):
                mlvl_keepid.append(keep_idxs)
                logits = cls_score[keep_idxs]       # H*W*PxClass
                mlvl_logits.append(logits)          # NxClass

        if kwargs.get('need_logits', False):
            kwargs['mlvl_logits'] = mlvl_logits
            kwargs['mlvl_keepid'] = mlvl_keepid

        results = self._bbox_post_process(mlvl_scores, mlvl_labels, mlvl_bboxes,
                                       img_meta['scale_factor'], cfg, rescale,
                                       with_nms, mlvl_score_factors, **kwargs)
        return results

    def _bbox_post_process(self,
                           mlvl_scores,
                           mlvl_labels,
                           mlvl_bboxes,
                           scale_factor,
                           cfg,
                           rescale=False,
                           with_nms=True,
                           mlvl_score_factors=None,
                           **kwargs):
        """bbox post-processing method.

        The boxes would be rescaled to the original image scale and do
        the nms operation. Usually `with_nms` is False is used for aug test.

        Args:
            mlvl_logits (list[Tensor]): Box logits from all scale
                levels of a single image, each item has shape
                (num_bboxes, num_classes).
            mlvl_scores (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_bboxes, ).
            mlvl_labels (list[Tensor]): Box class labels from all scale
                levels of a single image, each item has shape
                (num_bboxes, ).
            mlvl_bboxes (list[Tensor]): Decoded bboxes from all scale
                levels of a single image, each item has shape (num_bboxes, 4).
            scale_factor (ndarray, optional): Scale factor of the image arange
                as (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.
            mlvl_score_factors (list[Tensor], optional): Score factor from
                all scale levels of a single image, each item has shape
                (num_bboxes, ). Default: None.

        Returns:
            tuple[Tensor]: Results of detected bboxes and labels. If with_nms
                is False and mlvl_score_factor is None, return mlvl_bboxes and
                mlvl_scores, else return mlvl_bboxes, mlvl_scores and
                mlvl_score_factor. Usually with_nms is False is used for aug
                test. If with_nms is True, then return the following format

                - det_bboxes (Tensor): Predicted bboxes with shape \
                    [num_bboxes, 5], where the first 4 columns are bounding \
                    box positions (tl_x, tl_y, br_x, br_y) and the 5-th \
                    column are scores between 0 and 1.
                - det_labels (Tensor): Predicted labels of the corresponding \
                    box with shape [num_bboxes].
        """
        assert len(mlvl_scores) == len(mlvl_bboxes) == len(mlvl_labels)

        if kwargs.get('need_logits', False):
            mlvl_logits = kwargs['mlvl_logits']
            mlvl_keepid = kwargs['mlvl_keepid']
            assert len(mlvl_logits) == len(mlvl_keepid) == len(mlvl_labels)
            mlvl_logits = torch.cat(mlvl_logits)
            mlvl_keepid = torch.cat(mlvl_keepid)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        mlvl_labels = torch.cat(mlvl_labels)

        if mlvl_score_factors is not None:
            # TODO： Add sqrt operation in order to be consistent with
            #  the paper.
            mlvl_score_factors = torch.cat(mlvl_score_factors)
            mlvl_scores = mlvl_scores * mlvl_score_factors

        # ② PostNMS 后续对全部Level进行综合NMS，综合过滤
        if with_nms:
            if mlvl_bboxes.numel() == 0:
                det_bboxes = torch.cat([mlvl_bboxes, mlvl_scores[:, None]], -1)
                if kwargs.get('need_logits', False):
                    return det_bboxes, mlvl_labels, mlvl_logits, mlvl_keepid
                return det_bboxes, mlvl_labels

            det_bboxes, keep_idxs = batched_nms(mlvl_bboxes, mlvl_scores, mlvl_labels, cfg.nms)
            det_bboxes = det_bboxes[:cfg.max_per_img]
            det_labels = mlvl_labels[keep_idxs][:cfg.max_per_img]
            if kwargs.get('need_logits', False):
                # keep_idxs_list = keep_idxs.tolist()
                # mlvl_keepid_list = mlvl_keepid.tolist()
                # print('keep_idxs2=>', len(keep_idxs_list)==len(set(keep_idxs_list)), len(keep_idxs_list), len(set(keep_idxs_list)))
                # print('mlvl_keepid=>', len(mlvl_keepid_list)==len(set(mlvl_keepid_list)), len(mlvl_keepid_list), len(set(mlvl_keepid_list)))
                det_logits = mlvl_logits[keep_idxs][:cfg.max_per_img]
                det_keepid = mlvl_keepid[keep_idxs][:cfg.max_per_img]
                return det_bboxes, det_labels, det_logits, det_keepid
            return det_bboxes, det_labels
        else:
            if kwargs.get('need_logits', False):
                return mlvl_bboxes, mlvl_scores, mlvl_labels, mlvl_logits, mlvl_keepid
            else:
                return mlvl_bboxes, mlvl_scores, mlvl_labels

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