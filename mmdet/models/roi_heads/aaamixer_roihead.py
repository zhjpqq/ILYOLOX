import torch
from mmcv.runner import ModuleList
from torch import nn
from torch.nn import functional as F

from mmdet.core import bbox2result, bbox2roi, bbox_xyxy_to_cxcywh, build_assigner, build_sampler
from mmdet.core.bbox.samplers import PseudoSampler
from .bbox_heads.aaamixer_head import encode_box, decode_box
from ..builder import HEADS, build_head, build_roi_extractor, build_loss
from .cascade_roi_head import CascadeRoIHead
from .base_roi_head import BaseRoIHead
from .test_mixins import BBoxTestMixin, MaskTestMixin

import os
DEBUG = 'DEBUG' in os.environ


@HEADS.register_module()
class AaaMixerRoiHead(BaseRoIHead, BBoxTestMixin, MaskTestMixin):
    _DEBUG = -1

    def __init__(self,
                 num_stages=6,
                 stage_loss_weights=(1, 1, 1, 1, 1, 1),
                 content_dim=256,
                 featmap_strides=[4, 8, 16, 32],
                 bbox_roi_extractor=dict(
                     # This does not mean that our method need RoIAlign. We put this
                     # as a placeholder to satisfy the argument for the parent class
                     # 'CascadeRoIHead'.
                     type='SingleRoIExtractor',
                     roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=2),
                     out_channels=256,
                     featmap_strides=[4, 8, 16, 32]),
                 mask_roi_extractor=None,
                 bbox_head=None,
                 mask_head=None,
                 shared_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 has_teacher=False,
                 teacher_test_cfg=None,
                 feats_distill='',
                 loss_fd=dict(type='KnowledgeDistillationKLDivLoss', loss_weight=10, T=2),):
        assert bbox_head is not None
        assert len(stage_loss_weights) == num_stages
        self.has_teacher = has_teacher
        self.featmap_strides = featmap_strides
        self.num_stages = num_stages
        self.stage_loss_weights = stage_loss_weights
        self.content_dim = content_dim

        for i in range(num_stages):
            bbox_head[i].update(has_teacher=self.has_teacher)
            bbox_head[i].update(teacher_test_cfg=teacher_test_cfg)
        self.teacher_test_cfg = teacher_test_cfg

        # forked from cascade_roi_head.py
        assert bbox_roi_extractor is not None
        assert bbox_head is not None
        assert shared_head is None, 'Shared head is not supported in Cascade RCNN anymore'
        super(AaaMixerRoiHead, self).__init__(
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            mask_roi_extractor=mask_roi_extractor,
            mask_head=mask_head,
            shared_head=shared_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg)
        # train_cfg would be None when run the test.py
        if train_cfg is not None:
            for stage in range(num_stages):
                assert isinstance(self.bbox_sampler[stage], PseudoSampler)

        self.feats_distill = feats_distill
        self.loss_fd = build_loss(loss_fd) if feats_distill else None

    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        """Initialize box head and box roi extractor.
        Args:
            bbox_roi_extractor (dict): Config of box roi extractor.
            bbox_head (dict): Config of box in box head.
        """
        self.bbox_roi_extractor = ModuleList()
        self.bbox_head = ModuleList()
        if not isinstance(bbox_roi_extractor, list):
            bbox_roi_extractor = [bbox_roi_extractor for _ in range(self.num_stages)]
        if not isinstance(bbox_head, list):
            bbox_head = [bbox_head for _ in range(self.num_stages)]
        assert len(bbox_roi_extractor) == len(bbox_head) == self.num_stages
        for roi_extractor, head in zip(bbox_roi_extractor, bbox_head):
            self.bbox_roi_extractor.append(build_roi_extractor(roi_extractor))
            self.bbox_head.append(build_head(head))

    def init_mask_head(self, mask_roi_extractor, mask_head):
        """Initialize mask head and mask roi extractor.
        Args:
            mask_roi_extractor (dict): Config of mask roi extractor.
            mask_head (dict): Config of mask in mask head.
        """
        raise NotImplementedError("")

    def init_assigner_sampler(self):
        """Initialize assigner and sampler for each stage."""
        self.bbox_assigner = []
        self.bbox_sampler = []
        if self.train_cfg is not None:
            for idx, rcnn_train_cfg in enumerate(self.train_cfg):
                self.bbox_assigner.append(build_assigner(rcnn_train_cfg.assigner))
                self.current_stage = idx
                self.bbox_sampler.append(build_sampler(rcnn_train_cfg.sampler, context=self))

    def _bbox_forward(self, stage, img_feat, query_xyxy, query_content, imgs_whwh):
        num_imgs = imgs_whwh.size(0)

        cls_score, delta_score, delta_xyxy, query_content, cls_feat, loc_feat = self.bbox_head[stage].forward(
            img_feat, query_xyxy, query_content, self.featmap_strides)

        query_xyxy = self.bbox_head[stage].refine_xyxy(query_xyxy, delta_xyxy, imgs_whwh)

        decoded_bboxes = query_xyxy
        bboxes_list = [bboxes for bboxes in query_xyxy]

        bbox_results = dict(
            query_xyxy=query_xyxy,
            query_content=query_content,
            cls_feat=cls_feat,
            loc_feat=loc_feat,
            cls_score=cls_score,
            delta_score=delta_score,
            decode_bbox_pred=decoded_bboxes,
            detach_delta_score_list=[delta_score[i].detach() for i in range(num_imgs)] if delta_score is not None else [None]*num_imgs,
            detach_cls_score_list=[cls_score[i].detach() for i in range(num_imgs)],
            detach_bboxes_list=[item.detach() for item in bboxes_list],
            bboxes_list=bboxes_list)
        return bbox_results

    def forward_train(self,
                      x,
                      query_xyxy,
                      query_content,
                      img_metas,
                      gt_labels,
                      gt_bboxes,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      imgs_whwh=None,
                      semproto=None,
                      geoproto=None,
                      teacher_feat=None,
                      teacher_info=[],
                      task_labels={}):
        num_imgs = len(img_metas)
        num_query = query_xyxy.size(1)
        imgs_whwh = imgs_whwh.repeat(1, num_query, 1)
        all_stage_loss = {}

        for stage in range(self.num_stages):
            bbox_results = self._bbox_forward(stage, x, query_xyxy, query_content, imgs_whwh)
            if gt_bboxes_ignore is None:
                # TODO support ignore
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []

            cls_score_list = bbox_results['detach_cls_score_list']
            delta_score_list = bbox_results['detach_delta_score_list']
            bboxes_list = bbox_results['detach_bboxes_list']
            cls_score = bbox_results['cls_score']
            delta_score = bbox_results['delta_score']           # (N, n_query, 4*reg_num)
            decode_bbox_pred = bbox_results['decode_bbox_pred']
            query_bbox_prev = query_xyxy                    # (N, n_query, 4), prev box generated in prev stage!!
            cls_feat = bbox_results['cls_feat']
            loc_feat = bbox_results['loc_feat']

            if self.stage_loss_weights[stage] <= 0:
                continue

            for i in range(num_imgs):
                # 合并 GT-Label-Boxes & Teacher-Label-Boxes
                if self.has_teacher and stage == self.num_stages - 1 \
                        and stage in self.teacher_test_cfg.out_stage \
                        and 'hard' in self.bbox_head[stage].cates_distill:
                    gt_labels[i] = torch.cat([teacher_info[stage]['pred_label'][i], gt_labels[i]], dim=0)
                    gt_bboxes[i] = torch.cat([teacher_info[stage]['pred_bbox'][i], gt_bboxes[i]], dim=0)

                num_gtboxs = gt_bboxes[i].size(0)
                normalize_bbox_ccwh = bbox_xyxy_to_cxcywh(bboxes_list[i] / imgs_whwh[i])
                if num_gtboxs == 0:
                    continue

                if self.bbox_head[0].use_dfl:
                    reg_val = self.bbox_head[0].reg_val
                    bbox_prev = query_bbox_prev[i, :]
                    bbox_whwh = torch.cat([bbox_prev[..., 2:4] - bbox_prev[..., 0:2],
                                           bbox_prev[..., 2:4] - bbox_prev[..., 0:2]], dim=-1)
                    bbox_whwh = bbox_whwh[:, None, :].repeat(1, num_gtboxs, 1)
                    # bbox_whwh = imgs_whwh[i][:, None, :].repeat(1, num_gtboxs, 1)
                    delta_taget = gt_bboxes[i].repeat(num_query, 1) - bbox_prev.repeat(num_gtboxs, 1)
                    delta_taget = delta_taget.view(num_query, num_gtboxs, 4) / bbox_whwh
                    delta_taget = delta_taget[:, :, :, None].repeat(1, 1, 1, reg_val['num'])
                    delta_space = torch.linspace(reg_val['min'], reg_val['max'], reg_val['num'])
                    delta_space = delta_space.to(delta_taget.device).view(1, 1, 1, reg_val['num']).repeat(num_query, num_gtboxs, 4, 1)
                    delta_taget = (delta_taget - delta_space).abs().min(dim=-1)[1]  # => (nquery, ngtbox, 4)
                else:
                    delta_taget = None
                assign_result = self.bbox_assigner[stage].assign(
                    normalize_bbox_ccwh, cls_score_list[i], delta_score_list[i],
                    gt_bboxes[i], gt_labels[i], delta_taget, img_metas[i])
                # else:  # org assign
                #     assign_result = self.bbox_assigner[stage].assign(
                #         normalize_bbox_ccwh, cls_score_list[i], gt_bboxes[i], gt_labels[i], img_metas[i])

                sampling_result = self.bbox_sampler[stage].sample(assign_result, bboxes_list[i], gt_bboxes[i])
                sampling_results.append(sampling_result)

            label_bbox_targets = self.bbox_head[stage].get_targets(
                sampling_results, gt_bboxes, gt_labels, self.train_cfg[stage], True)

            single_stage_loss = self.bbox_head[stage].loss(
                cls_score.view(-1, cls_score.size(-1)),
                decode_bbox_pred.view(-1, 4),
                delta_score.view(-1, delta_score.size(-1)) if delta_score is not None else None,
                query_bbox_prev.view(-1, query_bbox_prev.size(-1)) if delta_score is not None else None,
                *label_bbox_targets,
                imgs_whwh=imgs_whwh,
                semproto=semproto,
                geoproto=geoproto,
                cls_feat=cls_feat,
                loc_feat=loc_feat,
                teacher_info=teacher_info[stage],
                task_labels=task_labels,
            )
            for key, value in single_stage_loss.items():
                all_stage_loss[f'stage{stage}_{key}'] = value * self.stage_loss_weights[stage]

            query_xyxy = bbox_results['query_xyxy'].detach()
            query_content = bbox_results['query_content']

        if self.has_teacher and 'kldv' in self.feats_distill:
            student_feat = x
            # print('feat identy = ', [round((sf == tf).nonzero().numel()/4/sf.numel(), 4)
            #                          for sf, tf in zip(student_feat, teacher_feat)])
            loss_fd = [self.loss_fd(sf, tf, weight=None, avg_factor=None)
                       for sf, tf in zip(student_feat, teacher_feat)]
            avg_factor = [1, len(loss_fd), len(img_metas)][2]
            loss_fd = sum(loss_fd) / avg_factor
            all_stage_loss.update({'loss_neck_fd': loss_fd})

        return all_stage_loss

    def simple_test(self,
                    x,
                    query_xyxy,
                    query_content,
                    img_metas,
                    imgs_whwh,
                    rescale=False,
                    cfg=None):
        assert self.with_bbox, 'Bbox head must be implemented.'

        num_imgs = len(img_metas)
        num_query = query_xyxy.size(1)
        imgs_whwh = imgs_whwh.repeat(1, num_query, 1)

        for stage in range(self.num_stages):
            bbox_results = self._bbox_forward(stage, x, query_xyxy, query_content, imgs_whwh)
            query_xyxy = bbox_results['query_xyxy']
            query_content = bbox_results['query_content']
            cls_score = bbox_results['cls_score']
            bboxes_list = bbox_results['detach_bboxes_list']

        num_classes = self.bbox_head[-1].num_classes
        det_bboxes = []
        det_labels = []

        if self.bbox_head[-1].loss_cls.use_sigmoid:
            cls_score = cls_score.sigmoid()
        else:
            cls_score = cls_score.softmax(-1)[..., :-1]

        for img_id in range(num_imgs):
            if self.test_cfg.filter == 'score':
                max_score, label = cls_score[img_id].max(dim=-1, keepdim=False)
                keepindex = max_score >= self.test_cfg.score_thr[stage]
                scores_per_img = max_score[keepindex]
                labels_per_img = label[keepindex]
                bbox_pred_per_img = bboxes_list[img_id][keepindex]
            elif self.test_cfg.filter == 'maxper':
                scores_per_img, topk_indices = cls_score[img_id].flatten(0, 1).topk(
                    self.test_cfg.max_per_img, sorted=False)
                labels_per_img = topk_indices % num_classes
                bbox_pred_per_img = bboxes_list[img_id][topk_indices // num_classes]
            else:
                raise NotImplementedError(f'cfg.filter={cfg.filter}')
            if rescale:
                scale_factor = img_metas[img_id]['scale_factor']
                bbox_pred_per_img /= bbox_pred_per_img.new_tensor(scale_factor)
            bbox_score_per_img = torch.cat([bbox_pred_per_img, scores_per_img[:, None]], dim=1)
            det_bboxes.append(bbox_score_per_img)
            det_labels.append(labels_per_img)

        bbox_results = [bbox2result(det_bboxes[i], det_labels[i], num_classes)
                        for i in range(num_imgs)]

        return bbox_results

    def complex_test(self,
                    x,
                    query_xyxy,
                    query_content,
                    img_metas,
                    imgs_whwh,
                    rescale=False):
        cfg = self.teacher_test_cfg
        num_imgs = len(img_metas)
        num_query = query_xyxy.size(1)
        imgs_whwh = imgs_whwh.repeat(1, num_query, 1)
        # assert self.with_bbox, 'Bbox head must be implemented.'
        # assert len(cfg.out_stage) <= self.num_stages, '输出stage不能大于总stage.'
        # assert len(cfg.score_thr) <= self.num_stages, '输出score_thr不能大于总stage.'

        bbox_result_list = []
        for stage in range(self.num_stages):
            bbox_results = self._bbox_forward(stage, x, query_xyxy, query_content, imgs_whwh)
            query_xyxy = bbox_results['query_xyxy']
            query_content = bbox_results['query_content']
            bbox_result_list.append(bbox_results)

        num_classes = self.bbox_head[-1].num_classes
        use_sigmoid = self.bbox_head[-1].loss_cls.use_sigmoid
        which_index = self.bbox_head[-1].which_index

        stage_result_list = []
        for stage in range(self.num_stages):
            if stage not in cfg.out_stage:
                stage_result_list.append(None)
                continue

            cls_score = bbox_result_list[stage]['cls_score']
            delta_score = bbox_result_list[stage]['delta_score']
            bbox_pred = bbox_result_list[stage]['decode_bbox_pred']

            cls_score = cls_score.sigmoid() if use_sigmoid else cls_score.softmax(-1)[..., :-1]

            det_scores, det_deltas, det_labels, det_bboxes, det_keepid, det_nums = [], [], [], [], [], []
            for img_id in range(num_imgs):
                if cfg.filter in ['score', 'score+maxper']:
                    max_score, label = cls_score[img_id].max(dim=-1, keepdim=False)
                    keepindex = max_score >= (cfg.score_thr[stage] if 'keepindex' in which_index else 0)
                    label_pred_per_img = label[keepindex]
                    bbox_pred_per_img = bbox_pred[img_id][keepindex]
                    cls_score_per_img = cls_score[img_id][keepindex]
                    delta_score_per_img = delta_score[img_id][keepindex]
                    if 'maxper' in cfg.filter:
                        raise NotImplementedError(f'cfg.filter={cfg.filter}')
                elif cfg.filter in ['maxper', 'maxper+score']:
                    # # maxper 筛选过滤   # 会打乱cls_score.size(0)中的原始顺序
                    scores_per_img, topk_indices = cls_score[img_id].flatten(0, 1).topk(cfg.max_per_img, sorted=False)
                    label_pred_per_img = topk_indices % num_classes
                    bbox_pred_per_img = bbox_pred[img_id][topk_indices // num_classes]
                    cls_score_per_img = cls_score[img_id][topk_indices // num_classes]
                    delta_score_per_img = delta_score[img_id][topk_indices // num_classes]
                    keepindex = scores_per_img >= 0.
                    # # score 筛选过滤
                    if 'score' in cfg.filter:
                        keepindex = scores_per_img > (cfg.score_thr[stage] if 'keepindex' in which_index else 0)
                        cls_score_per_img = cls_score_per_img[keepindex]
                        delta_score_per_img = delta_score_per_img[keepindex]
                        label_pred_per_img = label_pred_per_img[keepindex]
                        bbox_pred_per_img = bbox_pred_per_img[keepindex]
                else:
                    raise NotImplementedError(f'cfg.filter={cfg.filter}')

                if rescale:
                    scale_factor = img_metas[img_id]['scale_factor']
                    bbox_pred_per_img /= bbox_pred_per_img.new_tensor(scale_factor)
                det_scores.append(cls_score_per_img.detach())
                det_deltas.append(delta_score_per_img.detach())
                det_labels.append(label_pred_per_img.detach())
                det_bboxes.append(bbox_pred_per_img.detach())
                det_keepid.append(keepindex.detach())
                det_nums.append(label_pred_per_img.size(0))

            stage_result = {
                'det_nums': det_nums,           # 各图片检测数量
                'cls_score': det_scores,        # torch.cat(det_scores, dim=0),
                'delta_score': det_deltas,      # torch.cat(det_deltas, dim=0),
                'pred_label': det_labels,       # torch.cat(det_labels, dim=0),
                'pred_bbox': det_bboxes,        # torch.cat(det_bboxes, dim=0),
                'keepindex': det_keepid,        # torch.cat(det_keepid, dim=0),
            }
            stage_result_list.append(stage_result)

        return stage_result_list

    def aug_test(self, x, bboxes_list, img_metas, rescale=False):
        raise NotImplementedError()

    def forward_dummy(self, x,
                      query_xyzr,
                      query_content,
                      img_metas):
        """Dummy forward function when do the flops computing."""
        all_stage_bbox_results = []

        num_imgs = len(img_metas)
        if self.with_bbox:
            for stage in range(self.num_stages):
                bbox_results = self._bbox_forward(stage, x, query_xyzr, query_content, img_metas)
                all_stage_bbox_results.append(bbox_results)
                query_content = bbox_results['query_content']
                query_xyzr = bbox_results['query_xyzr']
        return all_stage_bbox_results
