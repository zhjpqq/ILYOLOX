import os
import random

import torch
import torch.nn as nn
from mmdet.models.losses.focal_loss import py_focal_loss_with_prob
from torch.nn import functional as F
from mmcv.cnn import (bias_init_with_prob, build_activation_layer, build_norm_layer)
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmcv.runner import auto_fp16, force_fp32

from mmdet.core import bbox_overlaps
from mmdet.core import multi_apply
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.dense_heads.atss_head import reduce_mean
from mmdet.models.losses import accuracy
from .aaaptive_mixing import AdaptiveSamplingMixing
from .bbox_head import BBoxHead
from mmdet.models.utils.misc import Integral
from mmdet.models.losses.utils import weight_reduce_loss as reduce_loss
from mmcv.ops import sigmoid_focal_loss
from mmdet.models.utils.misc import myactivate, gaussian

DEBUG = 'DEBUG' in os.environ


@HEADS.register_module()
class AaaMixerHead(BBoxHead):
    _DEBUG = -1

    def __init__(self,
                 num_classes=80,
                 num_ffn_fcs=2,
                 num_heads=8,
                 num_cls_fcs=1,
                 num_reg_fcs=1,
                 feedforward_channels=2048,
                 content_dim=256,
                 feat_channels=256,
                 dropout=0.0,
                 ffn_act_cfg=dict(type='ReLU', inplace=True),
                 in_points=32,
                 out_points=128,
                 n_groups=4,
                 loss_cls=dict(type='FocalLoss', use_sigmoid=True, activated=True, gamma=2.0, alpha=0.25, loss_weight=2),
                 loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
                 loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                 loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25),
                 init_cfg=None,
                 stage_id=None,

                 # prototyoe learning
                 prototype='',
                 loss_sempcenter=dict(type='KnowledgeDistillationKLDivLoss', loss_weight=10, T=2),
                 loss_sempsamper=dict(type='KnowledgeDistillationKLDivLoss', loss_weight=10, T=2),
                 loss_geoproto=dict(type='KnowledgeDistillationKLDivLoss', loss_weight=10, T=2),

                 # add for il
                 has_teacher=False,
                 teacher_test_cfg=None,
                 cross_xfl=False,
                 learn_fgbg='foreback',
                 hybrid_score=False,
                 mixed_clsloc=False,
                 which_index=False,
                 alpha_distill=1.,
                 loss_distill='',
                 cates_distill='',
                 locat_distill='',
                 feats_distill='',
                 reg_val={'min': -1, 'max': 1, 'num': 20, 'activate': 'sigmoid', 'method': 'v2', 'usedfl': True},
                 loss_cd_soft=dict(type='KnowledgeDistillationKLDivLoss', loss_weight=10, T=2),
                 loss_ld_bbox=dict(type='SmoothL1Loss', loss_weight=10, reduction='mean'),
                 loss_ld_iou=dict(type='GIoULoss', loss_weight=2.0, reduction='mean'),
                 loss_ld_soft=dict(type='KnowledgeDistillationKLDivLoss', loss_weight=0.25, T=10),
                 loss_fd=dict(type='KnowledgeDistillationKLDivLoss', loss_weight=10, T=2),
                 **kwargs):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
                                 'behavior, init_cfg is not allowed to be set'
        super(AaaMixerHead, self).__init__(
            num_classes=num_classes,
            reg_decoded_bbox=True,
            reg_class_agnostic=True,
            init_cfg=init_cfg,
            **kwargs)

        self.has_teacher = has_teacher
        self.teacher_test_cfg = teacher_test_cfg
        self.reg_val = reg_val
        self.use_dfl = self.reg_val['usedfl']
        self.num_location = 4 * self.reg_val['num'] if self.use_dfl else 4
        self.prototype = prototype
        self.cross_xfl = cross_xfl
        self.learn_fgbg = learn_fgbg
        self.hybrid_score = hybrid_score
        self.mixed_clsloc = mixed_clsloc
        self.which_index = which_index
        self.alpha_distill = alpha_distill

        self.stage_id = stage_id
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_iou = build_loss(loss_iou)
        self.content_dim = content_dim
        self.fp16_enabled = False
        self.attention = MultiheadAttention(content_dim, num_heads, dropout)
        self.attention_norm = build_norm_layer(dict(type='LN'), content_dim)[1]

        self.instance_interactive_conv_dropout = nn.Dropout(dropout)
        self.instance_interactive_conv_norm = build_norm_layer(
            dict(type='LN'), content_dim)[1]

        self.ffn = FFN(
            content_dim,
            feedforward_channels,
            num_ffn_fcs,
            act_cfg=ffn_act_cfg,
            dropout=dropout)
        self.ffn_norm = build_norm_layer(dict(type='LN'), content_dim)[1]

        self.cls_fcs = nn.ModuleList()
        for _ in range(num_cls_fcs):
            self.cls_fcs.append(nn.Linear(content_dim, content_dim, bias=True))
            self.cls_fcs.append(build_norm_layer(dict(type='LN'), content_dim)[1])
            self.cls_fcs.append(build_activation_layer(dict(type='ReLU', inplace=True)))

        # over load the self.fc_cls in BBoxHead
        if self.loss_cls.use_sigmoid:
            self.fc_cls = nn.Linear(content_dim, self.num_classes)
        else:
            self.fc_cls = nn.Linear(content_dim, self.num_classes + 1)              # TODO? why+1

        self.reg_fcs = nn.ModuleList()
        for _ in range(num_reg_fcs):
            self.reg_fcs.append(nn.Linear(content_dim, content_dim, bias=True))
            self.reg_fcs.append(build_norm_layer(dict(type='LN'), content_dim)[1])
            self.reg_fcs.append(build_activation_layer(dict(type='ReLU', inplace=True)))
        # over load the self.fc_cls in BBoxHead
        self.fc_reg = nn.Linear(content_dim, self.num_location)

        self.in_points = in_points
        self.n_heads = n_groups
        self.out_points = out_points

        self.sampling_n_mixing = AdaptiveSamplingMixing(
            content_dim=content_dim,  # query dim
            feat_channels=feat_channels,
            in_points=self.in_points,
            out_points=self.out_points,
            n_groups=self.n_heads)

        self.iof_tau = nn.Parameter(torch.ones(self.attention.num_heads, ))

        if self.use_dfl:
            self.integral = Integral(reg_val)
            self.loss_dfl = build_loss(loss_dfl)
        self.loss_distill = loss_distill
        self.cates_distill = cates_distill
        self.locat_distill = locat_distill
        self.feats_distill = feats_distill
        self.loss_cd_soft = build_loss(loss_cd_soft) #if cates_distill else loss_cd_soft
        self.loss_ld_bbox = build_loss(loss_ld_bbox) #if 'bbox' in locat_distill else None
        self.loss_ld_iou = build_loss(loss_ld_iou) #if 'iou' in locat_distill else None
        self.loss_ld_soft = build_loss(loss_ld_soft) #if 'soft' in locat_distill else None
        self.loss_fd = build_loss(loss_fd) #if feats_distill else None

        if 'semproto' in self.prototype:
            self.loss_sempcenter = build_loss(loss_sempcenter)
            self.loss_sempsamper = build_loss(loss_sempsamper)
            # 生成高斯正交特征监督标签
            gauss_eyes = torch.zeros(self.num_classes, content_dim)
            gseye_xxxx = torch.linspace(-1, 1, 2 * self.num_classes)
            gseye_xxxx = gaussian(gseye_xxxx, mean=0, std=0.5)
            for i in range(self.num_classes):
                start, end = self.num_classes - i, 2 * self.num_classes - i
                gauss_eyes[i] = gseye_xxxx[start: end]
                gauss_eyes[i] /= gauss_eyes.max(-1, keepdim=True)[0]
            self.gauss_eyes = gauss_eyes.detach()
        if 'geoproto' in self.prototype:
            self.loss_geoproto = build_loss(loss_geoproto)

    @torch.no_grad()
    def init_weights(self):
        super(AaaMixerHead, self).init_weights()
        for n, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                # adopt the default initialization for
                # the weight and bias of the layer norm
                pass
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            nn.init.constant_(self.fc_cls.bias, bias_init)

        nn.init.zeros_(self.fc_reg.weight)
        nn.init.zeros_(self.fc_reg.bias)

        nn.init.uniform_(self.iof_tau, 0.0, 4.0)

        self.sampling_n_mixing.init_weights()

    @auto_fp16()
    def forward(self,
                x,
                query_xyxy,
                query_content,
                featmap_strides):
        # print(f'\n====={self.stage_id}=======')
        N, n_query = query_content.shape[:2]

        query_xyzr = encode_box(query_xyxy)

        with torch.no_grad():
            # rois = decode_box(query_xyzr)
            rois = query_xyxy
            roi_box_batched = rois.view(N, n_query, 4)
            iof = bbox_overlaps(roi_box_batched, roi_box_batched, mode='iof')[:, None, :, :]
            iof = (iof + 1e-7).log()
            pe = position_embedding(query_xyzr, query_content.size(-1) // 4)

        '''IoF'''
        attn_bias = (iof * self.iof_tau.view(1, -1, 1, 1)).flatten(0, 1)

        query_content = query_content.permute(1, 0, 2)
        pe = pe.permute(1, 0, 2)
        '''sinusoidal positional embedding'''
        query_content_attn = query_content + pe
        query_content = self.attention(
            query_content_attn,
            attn_mask=attn_bias,
        )
        query_content = self.attention_norm(query_content)
        query_content = query_content.permute(1, 0, 2)      # => (N, n_query, 256)

        ''' adaptive 3D sampling and mixing '''
        query_content = self.sampling_n_mixing(x, query_content, query_xyzr, featmap_strides)

        # FFN
        query_content = self.ffn_norm(self.ffn(query_content))      # (N, n_query, 256)

        cls_feat = query_content
        reg_feat = query_content

        for cls_layer in self.cls_fcs:
            cls_feat = cls_layer(cls_feat)
        for reg_layer in self.reg_fcs:
            reg_feat = reg_layer(reg_feat)

        cls_score = self.fc_cls(cls_feat).view(N, n_query, -1)           # (N, n_query, 80)
        delta_score = self.fc_reg(reg_feat).view(N, n_query, -1)         # (N, n_query, 4*reg_num)
        if self.use_dfl:
            delta_xyxy = self.integral(delta_score, keepdim=True, stage=self.stage_id)
        else:
            delta_xyxy, delta_score = delta_score, None
        delta_xyxy = delta_xyxy.view(N, n_query, -1)

        cls_feat = cls_feat if 'semproto' in self.prototype else None
        reg_feat = reg_feat if 'geoproto' in self.prototype else None

        return cls_score, delta_score, delta_xyxy, query_content.view(N, n_query, -1), cls_feat, reg_feat

    def refine_xyzr(self, xyzr, xyzr_delta, return_bbox=True):
        # z=(h*w).sqrt().log2()
        # xyzr_delta[..., 0:2]=[d(x/z), d(y/z)]
        z = xyzr[..., 2:3]
        new_xy = xyzr[..., 0:2] + xyzr_delta[..., 0:2] * (2 ** z)
        new_zr = xyzr[..., 2:4] + xyzr_delta[..., 2:4]
        xyzr = torch.cat([new_xy, new_zr], dim=-1)
        if return_bbox:
            return xyzr, decode_box(xyzr)
        else:
            return xyzr

    def refine_xyxy(self, query_xyxy, delta_xyxy, imgs_whwh):
        # -1 <= -x/W <= x_delta <= 1-x/W <= 1  BxnumQx4
        wh = query_xyxy[..., 2:4] - query_xyxy[..., 0:2]
        whwh = torch.cat([wh, wh], dim=-1)
        # print(f'\n====={self.stage_id}=======')
        # print(f'delta={delta_xyxy[0][:8, :]}')
        # # # print(f'delta={whwh[0][:6, :]}')
        # # # print(f'delta*whwh={(delta_xyxy * whwh)[0][:6, :]}')
        # print(f'delta.min.max.sum={(delta_xyxy[0]).min()}, {(delta_xyxy[0]).max()}, {(delta_xyxy.flatten().abs()>=1).nonzero().numel()}')
        # print(f'delta*whwh.min.max={((delta_xyxy * whwh)[0][:, :]).min()}, {((delta_xyxy * whwh)[0][:, :]).max()}')
        query_xyxy = query_xyxy + delta_xyxy * whwh
        # 以下方式无法收敛
        # delta_xyxy = 2 * delta_xyxy.sigmoid() - 1
        # query_xyxy = query_xyxy + delta_xyxy * imgs_whwh
        return query_xyxy

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self,
             cls_score,
             bbox_pred,
             delta_score,
             bbox_prev,
             label_targets,
             label_weights,
             bbox_targets,
             bbox_weights,
             imgs_whwh=None,
             semproto=None,
             geoproto=None,
             cls_feat=None,
             loc_feat=None,
             reduction_override=None,
             teacher_info={},
             task_labels={},
             **kwargs):
        losses = dict()
        bg_class_ind = self.num_classes
        
        if teacher_info is not None and self.which_index in ['posandkeep', 'posnokeep']:
            # Hungry+Teacher双匹配, 消除LossTarget歧义!
            keepindex = torch.cat(teacher_info['keepindex'], dim=0)
            pred_label = torch.cat(teacher_info['pred_label'], dim=0)
            if 'posandkeep' in self.which_index:       # 只消除Hungry与Teacher的位置歧义！
                label_targets[keepindex] = pred_label
            elif 'posnokeep' in self.which_index:      # 同时消除Hungry与Teacher的位置歧义和语义歧义！
                label_targets[keepindex] = -1
            elif 'posonly' in self.which_index:
                pass                                   # 不根据keepindex对posindex做任何过滤！

        if 'posnoprev' in self.which_index:
            for label in task_labels['prev']:
                label_weights[label_targets == label] = 0
                bbox_weights[label_targets == label, ...] = 0
            # print(f"prev_labels ==> {task_labels['prev']}, label_weights ==> {label_weights}, {label_weights.min()}")

        pos_inds = (label_targets >= 0) & (label_targets < bg_class_ind)
        posindex = pos_inds.type(torch.bool)
        num_box = pos_inds.size(0)
        num_pos = pos_inds.sum().int().item()
        avg_factor = label_weights[posindex].sum().int().item()     # org num_pos
        imgs_whwh = imgs_whwh.reshape(num_box, 4)

        if cls_score is None or bbox_pred is None:
            losses.update({'loss_cls': cls_score.sum() * 0, 'loss_bbox': bbox_pred.sum() * 0,
                           'loss_iou': bbox_pred.sum() * 0, 'loss_dfl': bbox_pred.sum() * 0})
            print("=============raise NotImplementedError===========")
            return losses

        # assert bbox_pred.size(0) == num_box and bbox_pred.dim() == 2
        # bbox_pred = bbox_pred.reshape(bbox_pred.size(0), 4)

        # 直接传入score=Logits进行损失计算，不对Score进行Sigmoid()预处理使其成为Prob   # TODO 去掉背景类的学习
        if self.learn_fgbg == 'foreground':
            focal = sigmoid_focal_loss(cls_score[posindex], label_targets[posindex], self.loss_cls.gamma, self.loss_cls.alpha, None, 'none')
            label_weights = label_weights.view(-1, 1)[posindex]
        else:
            focal = sigmoid_focal_loss(cls_score, label_targets, self.loss_cls.gamma, self.loss_cls.alpha, None, 'none')
            label_weights = label_weights.view(-1, 1)
        xgiou = 1 - bbox_overlaps(bbox_pred[posindex], bbox_targets[posindex], mode='giou', is_aligned=True, eps=1e-7)
        l1dist = torch.abs(bbox_pred[posindex]/imgs_whwh[posindex] - bbox_targets[posindex]/imgs_whwh[posindex])

        if self.cross_xfl:
            fval, xval, lval = focal[posindex].sum(-1), xgiou,  l1dist.sum(-1)
            fxlsum = fval + xval + lval
            label_weights[posindex] = label_weights[posindex] * (1 + (xval + lval) / fxlsum)
            bbox_weights[posindex] = bbox_weights[posindex] * (1 + fval / fxlsum).view(-1, 1).repeat(1, 4)

        loss_cls = self.loss_cls.loss_weight * reduce_loss(focal, label_weights, self.loss_cls.reduction, avg_factor)
        loss_iou = self.loss_iou.loss_weight * reduce_loss(xgiou, bbox_weights[posindex].mean(-1), self.loss_iou.reduction, avg_factor)
        loss_bbox = self.loss_bbox.loss_weight * reduce_loss(l1dist, bbox_weights[posindex], self.loss_bbox.reduction, avg_factor)
        loss_iou = loss_iou * 0 if loss_iou < 0 else loss_iou   # TODO 防止IOU损失<0！

        loss_dfl = bbox_pred.sum() * 0
        if self.use_dfl:
            loss_dfl = self.loss_delta(delta_score, bbox_prev, bbox_targets, bbox_weights, imgs_whwh, posindex, avg_factor, num_box, num_pos)

        losses.update({'loss_cls': loss_cls, 'loss_bbox': loss_bbox, 'loss_iou': loss_iou, 'loss_dfl': loss_dfl})

        if self.prototype:
            loss_proto = self.loss_prototype(cls_feat, loc_feat, semproto, geoproto, posindex, label_targets, task_labels)
            losses.update(loss_proto)

        if self.has_teacher and teacher_info is not None:
            if self.loss_distill == 'v1':
                loss_distill = self.loss_distill_v1(teacher_info, cls_score, delta_score, bbox_pred, posindex, imgs_whwh)
            elif self.loss_distill == 'v2':
                loss_distill = self.loss_distill_v2(teacher_info, cls_score, delta_score, bbox_pred, posindex, imgs_whwh)
            else:
                raise NotImplementedError(f'{self.loss_distill}')
            loss_distill = {k: v * self.alpha_distill for k, v in loss_distill.items()}
            losses.update(loss_distill)

        return losses

    def loss_delta(self, delta_score, bbox_prev, bbox_targets, bbox_weights, imgs_whwh,
                   posindex, avg_factor, num_box, num_pos, reduction_override=None):
        bbox_whwh = torch.cat([bbox_prev[..., 2:4] - bbox_prev[..., 0:2],
                               bbox_prev[..., 2:4] - bbox_prev[..., 0:2]], dim=-1)
        # bbox_whwh = imgs_whwh
        delta_taget = (bbox_targets - bbox_prev) / bbox_whwh
        delta_taget = delta_taget[:, :, None].repeat(1, 1, self.reg_val['num'])
        delta_space = torch.linspace(self.reg_val['min'], self.reg_val['max'], self.reg_val['num'])
        delta_space = delta_space.to(delta_taget.device).view(1, 1, self.reg_val['num']).repeat(num_box, 4, 1)
        delta_taget = (delta_taget - delta_space).abs().min(dim=-1, keepdim=False)[1]  # (num_box, 4, reg_num)
        # x = torch.randn(2， 5, 4, 20)
        # xz = (x[F.one_hot(x.min(-1, False)[1], x.size(-1)).bool()]).view(x.size()[:-1]) == x.min(-1, False)[0]
        delta_weight = None
        if self.loss_dfl._get_name() == 'CrossEntropyLoss':
            if self.loss_dfl.use_sigmoid:
                avg_factor = avg_factor  # * 4 * self.reg_val['num']
                delta_taget = F.one_hot(delta_taget, self.reg_val['num'])
                delta_taget = delta_taget[posindex].view(num_pos, 4, self.reg_val['num'])
                delta_score = delta_score[posindex].view(num_pos, 4, self.reg_val['num'])
                delta_score = myactivate(delta_score, func=self.loss_dfl.activate, dim=-1)  # ？？TODO logits
                loss_dfl = F.binary_cross_entropy_with_logits(
                    delta_score, delta_taget.float(), pos_weight=None, reduction='none')
                loss_dfl = self.loss_dfl.loss_weight * reduce_loss(
                    loss_dfl, delta_weight, self.loss_dfl.reduction, avg_factor)
            else:
                avg_factor = avg_factor # * 4 * self.reg_val['num']
                delta_taget = delta_taget[posindex].view(num_pos * 4)
                delta_score = delta_score[posindex].view(num_pos * 4, self.reg_val['num'])
                delta_score = myactivate(delta_score, func=self.loss_dfl.activate, dim=-1)   # ？？TODO logits
                loss_dfl = F.cross_entropy(delta_score, delta_taget, weight=None, reduction='none')
                loss_dfl = self.loss_dfl.loss_weight * reduce_loss(
                    loss_dfl, delta_weight, self.loss_dfl.reduction, avg_factor)
        elif self.loss_dfl._get_name() == 'FocalLoss':
            # assert self.reg_val['activate'] == 'sigmoid', 'softmax not supported! focal_loss.py Line243'
            # HunguryLine142:沿类别求和，再沿xyxy求和，在reduce_loss中一次性操作, 每次都需要4个坐标点都同时以最小损失匹配
            avg_factor = avg_factor * 1
            delta_weight = bbox_weights[posindex].view(num_pos * 4, 1)
            delta_taget = delta_taget[posindex].view(num_pos * 4)
            delta_score = delta_score[posindex].view(num_pos * 4, self.reg_val['num'])
            delta_score = myactivate(delta_score, func=self.reg_val['activate'], dim=-1)
            # if random.random() > 0.9995:
            #     delta_pred0, delta_pred1 = delta_score.max(-1, False)
            #     end = random.randint(0, num_pos)
            #     start = max(0, end-15)
            #     # start, end = 0, 15
            #     taget_pred_score = list(zip(delta_taget.view(-1, 4)[start:end, :].flatten().detach().cpu().numpy(),
            #                                 delta_pred1.view(-1, 4)[start:end, :].flatten().detach().cpu().numpy(),
            #                                 delta_pred0.view(-1, 4)[start:end, :].flatten().detach().cpu().numpy()))
            #     # delta_pred0.view(-1, 4)[start:end, :].flatten().detach().cpu().numpy()
            #     print(f'\nstage={self.stage_id} delta_target vs delta_pred1 vs delta_pred0\n',
            #           f'\ntaget_pred_score={taget_pred_score}',
            #           # f'\ndelta_taget={delta_taget.view(-1, 4)[start:end, :].flatten()}',
            #           # f'\ndelta_pred1={delta_pred1.view(-1, 4)[start:end, :].flatten()}',
            #           # f'\ndelta_pred0={delta_pred0.view(-1, 4)[start:end, :].flatten()}',
            #           '\n==========================================')
            focal = py_focal_loss_with_prob(
                delta_score, delta_taget, None, self.loss_dfl.gamma, self.loss_dfl.alpha, 'none', None)
            # print(f'avg_factor=>{avg_factor} focal=>{focal.size()}, '
            #       f'delta_weight=>{delta_weight.size()}, '
            #       f'fxd=>{(focal*delta_weight).size()}')
            loss_dfl = self.loss_dfl.loss_weight * reduce_loss(
                focal, delta_weight, self.loss_dfl.reduction, avg_factor)
        else:
            raise NotImplementedError(f'{self.loss_dfl._get_name()}')
        return loss_dfl

    def loss_distill_v2(self, teacher_info, cls_score, delta_score, bbox_pred, posindex, imgs_whwh):
        if 'keepindex' in self.which_index:
            keepidx = torch.cat(teacher_info['keepindex'], dim=0)
        else:
            keepidx = posindex
        keep_num = keepidx.nonzero().numel()
        reg_num = self.reg_val['num']
        cd_T, ld_T = getattr(self.loss_cd_soft, 'T', None), getattr(self.loss_ld_soft, 'T', None)
        losses = dict()
        ## loss()函数必须返回相同数量的loss项，否则GPU无法分布式同步
        # if keep_num == 0:  return losses

        if self.loss_cls.use_sigmoid:
            batch_cls_score = cls_score[keepidx].sigmoid()
            if self.loss_cd_soft._get_name() in ['KnowledgeDistillationKLDivLoss']:
                batch_cls_score = batch_cls_score / batch_cls_score.sum(-1, keepdim=True)     # x.norm_sigmoid()
        else:
            batch_cls_score = cls_score[keepidx].softmax(-1)[..., :-1]
        batch_delta_score = delta_score[keepidx].view(keep_num, 4, reg_num)
        batch_delta_score = myactivate(batch_delta_score, func=self.reg_val['activate'], dim=-1)
        batch_pred_bbox = bbox_pred[keepidx]

        soft_cls_score = torch.cat(teacher_info['cls_score'], dim=0)   # sigmoid already done in  complex_test()!
        if self.loss_cls.use_sigmoid and self.loss_cd_soft._get_name() in ['KnowledgeDistillationKLDivLoss']:
            soft_cls_score = soft_cls_score / soft_cls_score.sum(-1, keepdim=True)
        soft_delta_score = torch.cat(teacher_info['delta_score'], dim=0).view(-1, 4, reg_num)
        soft_delta_score = myactivate(soft_delta_score, func=self.reg_val['activate'], dim=-1)
        soft_pred_bbox = torch.cat(teacher_info['pred_bbox'], dim=0)

        if self.hybrid_score and keep_num > 0:
            # TODO 合并GT框后已经有onehot监督！？
            # x[F.one_hot(x.max(-1)[1], x.size(-1)).bool()].view(x.size()[:-1]) == x.max(-1)[0]
            soft_cls_score[F.one_hot(soft_cls_score.max(-1)[1], self.num_classes).bool()] = 1.0
            soft_delta_score[F.one_hot(soft_delta_score.max(-1)[1], reg_num).bool()] = 1.0
                
        # !!!温度T越大，F.kl_div()越小!!!
        # KLD https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html
        # c_kld = F.kl_div(
        #     F.log_softmax(batch_cls_score / cd_T, dim=-1),
        #     F.softmax(soft_cls_score / cd_T, dim=-1).detach(),
        #     reduction='none').mean(-1) * (cd_T * cd_T)
        # l_kld = F.kl_div(
        #     F.log_softmax(batch_delta_score / ld_T, dim=-1),
        #     F.softmax(soft_delta_score / ld_T, dim=-1).detach(),
        #     reduction='none').mean(-1) * (ld_T * ld_T)
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
        if self.loss_ld_soft._get_name() == 'KnowledgeDistillationKLDivLoss':
            loc_err = F.kl_div(
                (batch_delta_score.view(keep_num, 4 * reg_num) / ld_T).log(),
                (soft_delta_score.view(keep_num, 4 * reg_num) / ld_T).detach(),
                reduction='none').mean(-1) * (ld_T * ld_T)   # todo mean or sum
        elif self.loss_ld_soft._get_name() == 'FocalLoss':
            pred = batch_delta_score.view(keep_num, 4, reg_num)
            target = soft_delta_score.view(keep_num, 4, reg_num).type_as(pred)
            alpha, gamma = self.loss_ld_soft.alpha, self.loss_ld_soft.gamma
            pt = (1 - pred) * target + pred * (1 - target)
            focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * pt.pow(gamma)
            loc_err = F.binary_cross_entropy(pred, target, reduction='none') * focal_weight
            loc_err = loc_err.sum(-1).sum(-1)
        elif self.loss_ld_soft._get_name() == 'MSELoss':
            loc_err = F.mse_loss(
                batch_delta_score.view(keep_num, 4 * reg_num),
                soft_delta_score.view(keep_num, 4 * reg_num),
                reduction='none').sum(-1)
        else:
            raise NotImplementedError
        l1dist = torch.abs(batch_pred_bbox / imgs_whwh[keepidx] - soft_pred_bbox / imgs_whwh[keepidx])
        if self.loss_ld_bbox and self.loss_ld_bbox._get_name() == 'SmoothL1Loss':
            l1dist = torch.where(l1dist < 1.0, 0.5 * l1dist * l1dist / 1.0, l1dist - 0.5 * 1.0)
        xgiou = 1 - bbox_overlaps(batch_pred_bbox, soft_pred_bbox, mode='giou', is_aligned=True, eps=1e-7)

        cls_weight, loc_weight, bbox_weight, iou_weight = [None] * 4
        if self.mixed_clsloc and keep_num > 0:
            # N阶协同多任务损失函数；+为0阶，相互包含N次为N阶；
            # 连续学习下的分类/定位协同样本挖掘：共同样本挖掘：0~2之间，1处代表分类定位的平衡，
            # (cls, loc) = (坏,坏)，(坏,好)，(好,坏)，(好,好).
            cls_weight = 2 * cls_err / (cls_err + loc_err)
            loc_weight = 2 * loc_err / (cls_err + loc_err)

        if 'soft' in self.cates_distill:
            avg_factor = keep_num
            loss_cd_soft = self.loss_cd_soft.loss_weight * reduce_loss(cls_err, cls_weight, self.loss_cd_soft.reduction, avg_factor)
            # loss_cd_soft1 = self.loss_cd_soft(batch_cls_score, soft_cls_score, weight=cls_weight, avg_factor=avg_factor)
            # assert loss_cd_soft1 == loss_cd_soft
            losses.update({'loss_cd_soft': loss_cd_soft})

        if 'soft' in self.locat_distill:
            avg_factor = keep_num * 1
            loss_ld_soft = self.loss_ld_soft.loss_weight * reduce_loss(loc_err, loc_weight, self.loss_ld_soft.reduction, avg_factor)
            # loss_ld_soft1 = self.loss_ld_soft(batch_delta_score, soft_delta_score, weight=loc_weight, avg_factor=avg_factor)
            # assert loss_ld_soft1 == loss_ld_soft
            losses.update({'loss_ld_soft': loss_ld_soft})

        if 'bbox' in self.locat_distill:
            avg_factor = keep_num
            loss_ld_bbox = self.loss_ld_bbox.loss_weight * reduce_loss(l1dist, bbox_weight, self.loss_ld_bbox.reduction, avg_factor)
            # loss_ld_bbox1 = self.loss_ld_bbox(batch_pred_bbox / imgs_whwh[keepidx], soft_pred_bbox / imgs_whwh[keepidx], weight=bbox_weight, avg_factor=avg_factor)
            # assert loss_ld_bbox1 == loss_ld_bbox
            losses.update({'loss_ld_bbox': loss_ld_bbox})

        if 'iou' in self.locat_distill:
            avg_factor = keep_num
            loss_ld_iou = self.loss_ld_iou.loss_weight * reduce_loss(xgiou, iou_weight, self.loss_ld_iou.reduction, avg_factor)
            # loss_ld_iou1 = self.loss_ld_iou(batch_pred_bbox, soft_pred_bbox, weight=iou_weight, avg_factor=avg_factor)
            # assert loss_ld_iou1 == loss_ld_iou
            losses.update({'loss_ld_iou': loss_ld_iou})

        if keep_num == 0:
            assert all([x.item() == 0 for x in losses.values()]), print(f'keep_num={keep_num}, {losses}')

        return losses
        
    def loss_prototype(self, cls_feat, loc_feat, semproto, geoproto, posindex, label_target, task_labels={}):
        losses = dict()   # 按新旧任务区分

        if 'semproto' in self.prototype:
            # OMP正交匹配追踪 高斯正交优先原型特征 动态滤波原型特征
            loss_semp = label_target.sum() * 0
            N, n_query, feat_dim = cls_feat.size()
            # assert feat_dim == semproto.size(1)
            cls_feat = cls_feat.view(N * n_query, feat_dim)[posindex]
            label_idx = label_target[posindex]
            label_prev_task = set(task_labels['prev'])
            label_idx_new = torch.cat([l for l in label_idx if l not in label_prev_task], dim=0)

            # 计算新增类别的动态均值聚类中心作为原型特征，旧类别原型不再更新！
            # for l in set(label_idx.tolist()):
            #     if label_prev_task and l in label_prev_task:
            #         semproto[l] = semproto[l].detach()
            #         continue
            #     semproto[l] = (semproto[l] + cls_feat[label_idx == l].mean(dim=0)) / 2.
            for l in set(label_idx_new.tolist()):
                semproto[l] = (semproto[l] + cls_feat[label_idx == l].mean(dim=0)) / 2.

            # 计算新增类别原型正交损失
            sempcenter = F.mse_loss(semproto[label_idx_new], self.gauss_eyes[label_idx_new], reduction='none')

            # 计算所有类别原型样本损失
            sempsample = F.mse_loss(semproto[label_idx], cls_feat[label_idx], reduction='none')

            if 'semproto_center' in self.prototype:
                center_weight = None
                avg_factor = label_idx_new.numel()
                loss_sempcenter = self.loss_sempcenter.loss_weight * reduce_loss(
                    sempcenter, center_weight, self.loss_sempcenter.reduce, avg_factor)
                losses.update({'loss_sempcenter': loss_sempcenter})

            if 'semproto_sample' in self.prototype:
                sample_weight = None
                avg_factor = label_idx.numel()
                loss_sempsample = self.loss_sempsample.loss_weight * reduce_loss(
                    sempsample, sample_weight, self.loss_sempsample.reduce, avg_factor)
                losses.update({'loss_sempsample': loss_sempsample})

        if 'geoproto' in self.prototype:
            loss_geop = label_target.sum() * 0
            losses.update({'loss_geop': loss_geop})

        return losses

    def loss_distill_v1(self, teacher_info, cls_score, delta_score, bbox_pred, posindex, imgs_whwh):
        keepidx = torch.cat(teacher_info['keepindex'], dim=0)
        avg_factor = keepidx.nonzero().numel()
        reg_num = self.reg_val['num']
        losses = dict()

        if 'soft' in self.cates_distill and avg_factor > 0:
            if self.loss_cls.use_sigmoid:
                batch_cls_score = cls_score[keepidx].sigmoid()
            else:
                batch_cls_score = cls_score[keepidx].softmax(-1)[..., :-1]
            soft_cls_score = torch.cat(teacher_info['cls_score'], dim=0)
            # print(f'batch-soft cls-socre !=> {(batch_cls_score != soft_cls_score).nonzero().numel()}')
            # # hybrid soft class distill
            # soft_cls_score[F.one_hot(soft_cls_score.max(-1)[1], self.num_classes).bool()] == soft_cls_score.max(-1)[0]
            soft_cls_score[F.one_hot(soft_cls_score.max(-1)[1], self.num_classes).bool()] = 1.0
            loss_cd_soft = self.loss_cd_soft(batch_cls_score, soft_cls_score, weight=None, avg_factor=avg_factor)
            losses.update({'loss_cd_soft': loss_cd_soft})

        if 'soft' in self.locat_distill and avg_factor > 0:
            # forked from ld_head.py Line99-Line122
            batch_delta_score = delta_score[keepidx].view(-1, 4, reg_num)
            soft_delta_score = torch.cat(teacher_info['delta_score'], dim=0).view(-1, 4, reg_num)
            batch_delta_score = myactivate(batch_delta_score, func=self.reg_val['activate'], dim=-1)
            soft_delta_score = myactivate(soft_delta_score, func=self.reg_val['activate'], dim=-1)
            # print(f'batch-soft delta-score !=> {(batch_delta_score != soft_delta_score).nonzero().numel()}')
            # # hybrid soft location distill
            # soft_delta_score[F.one_hot(soft_delta_score.max(-1)[1], reg_num).bool()] == soft_delta_score.max(-1)[0].flatten()
            soft_delta_score[F.one_hot(soft_delta_score.max(-1)[1], reg_num).bool()] = 1.0
            loss_ld_soft = self.loss_ld_soft(batch_delta_score.view(-1, 4 * reg_num),
                                             soft_delta_score.view(-1, 4 * reg_num),
                                             weight=None, avg_factor=avg_factor)
            losses.update({'loss_ld_soft': loss_ld_soft})

        if 'bbox' in self.locat_distill and avg_factor > 0:
            # 按照keepidx匹配对进行的蒸馏，与合并Lable再GetTarget匹配进行的蒸馏略有不同
            batch_pred_bbox = bbox_pred[keepidx]
            soft_pred_bbox = torch.cat(teacher_info['pred_bbox'], dim=0)
            loss_ld_bbox = self.loss_ld_bbox(batch_pred_bbox, soft_pred_bbox, weight=None, avg_factor=avg_factor)
            # l1dist = torch.abs(batch_pred_bbox / imgs_whwh[keepidx] - soft_pred_bbox / imgs_whwh[keepidx])
            # loss_ld_bbox = self.loss_ld_bbox.loss_weight * reduce_loss(l1dist, None, self.loss_ld_bbox.reduction, avg_factor)
            losses.update({'loss_ld_bbox': loss_ld_bbox})

        if 'iou' in self.locat_distill and avg_factor > 0:
            batch_pred_bbox = bbox_pred[keepidx]
            soft_pred_bbox = torch.cat(teacher_info['pred_bbox'], dim=0)
            loss_ld_iou = self.loss_ld_iou(batch_pred_bbox, soft_pred_bbox, weight=None, avg_factor=avg_factor)
            # xgiou = 1 - bbox_overlaps(batch_pred_bbox, soft_pred_bbox, mode='giou', is_aligned=True, eps=1e-7)
            # loss_ld_iou = self.loss_ld_iou.loss_weight * reduce_loss(xgiou, None, self.loss_ld_iou.reduction, avg_factor)
            losses.update({'loss_ld_iou': loss_ld_iou})

        return losses

    def loss_distill_v3(self, losses, teacher_info, cls_score, delta_score, bbox_pred, imgs_whwh):
        keepidx = torch.cat(teacher_info['keepindex'], dim=0)
        avg_factor = keepidx.nonzero().numel()
        reg_num = self.reg_val['num']
        if avg_factor == 0:
            return losses

        if self.loss_cls.use_sigmoid:
            batch_cls_score = cls_score[keepidx].sigmoid()
        else:
            batch_cls_score = cls_score[keepidx].softmax(-1)[..., :-1]
        batch_delta_score = delta_score[keepidx].view(-1, 4, reg_num)
        batch_delta_score = myactivate(batch_delta_score, func=self.reg_val['activate'], dim=-1)
        batch_pred_bbox = bbox_pred[keepidx]

        soft_cls_score = torch.cat(teacher_info['cls_score'], dim=0)
        soft_delta_score = torch.cat(teacher_info['delta_score'], dim=0).view(-1, 4, reg_num)
        soft_delta_score = myactivate(soft_delta_score, func=self.reg_val['activate'], dim=-1)
        soft_pred_bbox = torch.cat(teacher_info['pred_bbox'], dim=0)

        if 'soft' in self.cates_distill:
            soft_cls_score[F.one_hot(soft_cls_score.max(-1)[1], self.num_classes).bool()] = 1.0
            loss_cd_soft = self.loss_cd_soft(batch_cls_score, soft_cls_score, weight=None, avg_factor=avg_factor)
            losses.update({'loss_cd_soft': loss_cd_soft})

        if 'soft' in self.locat_distill:
            soft_delta_score[F.one_hot(soft_delta_score.max(-1)[1], reg_num).bool()] = 1.0
            loss_ld_soft = self.loss_ld_soft(batch_delta_score.view(-1, 4 * reg_num),
                                             soft_delta_score.view(-1, 4 * reg_num),
                                             weight=None, avg_factor=avg_factor)
            losses.update({'loss_ld_soft': loss_ld_soft})

        if 'bbox' in self.locat_distill:
            # loss_ld_bbox = self.loss_ld_bbox(batch_pred_bbox, soft_pred_bbox, weight=None, avg_factor=avg_factor)
            l1dist = torch.abs(batch_pred_bbox / imgs_whwh[keepidx] - soft_pred_bbox / imgs_whwh[keepidx])
            l1dist = torch.where(l1dist < 1.0, 0.5 * l1dist * l1dist / 1.0, l1dist - 0.5 * 1.0)
            loss_ld_bbox = self.loss_ld_bbox.loss_weight * reduce_loss(l1dist, None, self.loss_ld_bbox.reduction, avg_factor)
            losses.update({'loss_ld_bbox': loss_ld_bbox})

        if 'iou' in self.locat_distill:
            # loss_ld_iou = self.loss_ld_iou(batch_pred_bbox, soft_pred_bbox, weight=None, avg_factor=avg_factor)
            xgiou = 1 - bbox_overlaps(batch_pred_bbox, soft_pred_bbox, mode='giou', is_aligned=True, eps=1e-7)
            loss_ld_iou = self.loss_ld_iou.loss_weight * reduce_loss(xgiou, None, self.loss_ld_iou.reduction, avg_factor)
            losses.update({'loss_ld_iou': loss_ld_iou})

        return losses

    def _get_target_single(self, pos_inds, neg_inds, pos_bboxes, neg_bboxes,
                           pos_gt_bboxes, pos_gt_labels, cfg):
        num_pos = pos_bboxes.size(0)
        num_neg = neg_bboxes.size(0)
        num_samples = num_pos + num_neg

        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]
        label_targets = pos_bboxes.new_full((num_samples,), self.num_classes, dtype=torch.long)
        label_weights = pos_bboxes.new_zeros(num_samples)
        bbox_targets = pos_bboxes.new_zeros(num_samples, 4)
        bbox_weights = pos_bboxes.new_zeros(num_samples, 4)
        if num_pos > 0:
            label_targets[pos_inds] = pos_gt_labels
            pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
            label_weights[pos_inds] = pos_weight
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(pos_bboxes, pos_gt_bboxes)
            else:
                pos_bbox_targets = pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1
        if num_neg > 0:
            label_weights[neg_inds] = 1.0

        return label_targets, label_weights, bbox_targets, bbox_weights

    def get_targets(self,
                    sampling_results,
                    gt_bboxes,
                    gt_labels,
                    rcnn_train_cfg,
                    concat=True):
        pos_inds_list = [res.pos_inds for res in sampling_results]
        neg_inds_list = [res.neg_inds for res in sampling_results]
        pos_bboxes_list = [res.pos_bboxes for res in sampling_results]
        neg_bboxes_list = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]
        label_targets, label_weights, bbox_targets, bbox_weights = multi_apply(
            self._get_target_single,
            pos_inds_list,
            neg_inds_list,
            pos_bboxes_list,
            neg_bboxes_list,
            pos_gt_bboxes_list,
            pos_gt_labels_list,
            cfg=rcnn_train_cfg)
        if concat:
            label_targets = torch.cat(label_targets, 0)
            label_weights = torch.cat(label_weights, 0)
            bbox_targets = torch.cat(bbox_targets, 0)
            bbox_weights = torch.cat(bbox_weights, 0)
        return label_targets, label_weights, bbox_targets, bbox_weights

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss_org(self,
                 cls_score,
                 bbox_pred,
                 delta_score,
                 bbox_prev,
                 label_targets,
                 label_weights,
                 bbox_targets,
                 bbox_weights,
                 imgs_whwh=None,
                 reduction_override=None,
                 **kwargs):
        losses = dict()
        bg_class_ind = self.num_classes

        pos_inds = (label_targets >= 0) & (label_targets < bg_class_ind)
        posindex = pos_inds.type(torch.bool)
        num_box = pos_inds.size(0)
        num_pos = pos_inds.sum().int().item()
        avg_factor = reduce_mean(pos_inds.sum().float())
        imgs_whwh = imgs_whwh.reshape(num_box, 4)

        assert bbox_pred.size(0) == num_box and bbox_pred.dim() == 2
        # bbox_pred = bbox_pred.reshape(bbox_pred.size(0), 4)

        if cls_score is not None:
            if cls_score.numel() > 0:
                losses['loss_cls'] = self.loss_cls(
                    cls_score,
                    label_targets,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                # losses['pos_acc'] = accuracy(cls_score[pos_inds], label_targets[pos_inds])

        if delta_score is not None:
            if pos_inds.any() and self.loss_dfl.loss_weight > 0:
                if self.loss_dfl._get_name():   # == 'CrossEntropyLoss':
                    bbox_whwh = torch.cat([bbox_prev[..., 2:4] - bbox_prev[..., 0:2],
                                           bbox_prev[..., 2:4] - bbox_prev[..., 0:2]], dim=-1)
                    delta_taget = (bbox_targets - bbox_prev)/bbox_whwh
                    delta_taget = delta_taget[:, :, None].repeat(1, 1, self.reg_val['num'])
                    delta_space = torch.linspace(self.reg_val['min'], self.reg_val['max'], self.reg_val['num'])
                    delta_space = delta_space.to(delta_taget.device).view(1, 1, self.reg_val['num']).repeat(num_box, 4, 1)
                    delta_taget = (delta_taget - delta_space).abs().min(dim=-1)[1]
                    delta_weight = None  # delta_label.ones_like(delta_label.size())

                    if self.loss_dfl.use_sigmoid:
                        delta_taget = F.one_hot(delta_taget, self.reg_val['num'])
                        losses['loss_dfl'] = self.loss_dfl(
                            delta_score[posindex].view(num_pos, 4, self.reg_val['num']).softmax(-1),
                            delta_taget[posindex].view(num_pos, 4, self.reg_val['num']),
                            delta_weight,
                            avg_factor=avg_factor * 4 * self.reg_val['num'],
                            reduction_override=reduction_override)
                    else:
                        losses['loss_dfl'] = self.loss_dfl(
                            delta_score[posindex].view(num_pos*4, self.reg_val['num']).softmax(-1),
                            delta_taget[posindex].view(num_pos*4),
                            delta_weight,
                            avg_factor=avg_factor * 4 * self.reg_val['num'],
                            reduction_override=reduction_override)
            else:
                losses['loss_dfl'] = delta_score.sum() * 0

        if bbox_pred is not None:
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                losses['loss_bbox'] = self.loss_bbox(
                    bbox_pred[posindex] / imgs_whwh[posindex],
                    bbox_targets[posindex] / imgs_whwh[posindex],
                    bbox_weights[posindex],
                    avg_factor=avg_factor)
                losses['loss_iou'] = self.loss_iou(
                    bbox_pred[posindex],
                    bbox_targets[posindex],
                    bbox_weights[posindex],
                    avg_factor=avg_factor)
            else:
                losses['loss_bbox'] = bbox_pred.sum() * 0
                losses['loss_iou'] = bbox_pred.sum() * 0

        return losses

    def loss_delta_org(self, delta_score, bbox_prev, bbox_targets, posindex,
                        avg_factor, num_box, num_pos, reduction_override=None):
        # 比较API是否正确
        loss_dfl = delta_score.sum() * 0
        if self.loss_dfl._get_name():  # == 'CrossEntropyLoss':
            bbox_whwh = torch.cat([bbox_prev[..., 2:4] - bbox_prev[..., 0:2],
                                   bbox_prev[..., 2:4] - bbox_prev[..., 0:2]], dim=-1)
            delta_taget = (bbox_targets - bbox_prev) / bbox_whwh
            delta_taget = delta_taget[:, :, None].repeat(1, 1, self.reg_val['num'])
            delta_space = torch.linspace(self.reg_val['min'], self.reg_val['max'], self.reg_val['num'])
            delta_space = delta_space.to(delta_taget.device).view(1, 1, self.reg_val['num']).repeat(num_box, 4, 1)
            delta_taget = (delta_taget - delta_space).abs().min(dim=-1)[1]
            delta_weight = None  # delta_label.ones_like(delta_label.size())

            if self.loss_dfl.use_sigmoid:
                avg_factor = avg_factor * 4 * self.reg_val['num']
                delta_taget = F.one_hot(delta_taget, self.reg_val['num'])
                delta_taget = delta_taget[posindex].view(num_pos, 4, self.reg_val['num'])
                delta_score = delta_score[posindex].view(num_pos, 4, self.reg_val['num'])
                delta_score = myactivate(delta_score, func=self.loss_dfl.activate, dim=-1)
                loss_dfl = self.loss_dfl(delta_score, delta_taget, delta_weight, avg_factor, reduction_override)
                # loss_dflx = F.binary_cross_entropy_with_logits(
                #     delta_score, delta_taget.float(), pos_weight=delta_weight, reduction='none')
                # loss_dflx = self.loss_dfl.loss_weight * reduce_loss(
                #     loss_dflx, delta_weight, self.loss_dfl.reduction, avg_factor)
                # print(f'loss_dflx = {loss_dfl} == {loss_dflx} ? {loss_dfl==loss_dflx}')
            else:
                avg_factor = avg_factor * 4 * self.reg_val['num']
                delta_taget = delta_taget[posindex].view(num_pos * 4)
                delta_score = delta_score[posindex].view(num_pos * 4, self.reg_val['num'])
                delta_score = myactivate(delta_score, func=self.loss_dfl.activate, dim=-1)
                loss_dfl = self.loss_dfl(delta_score, delta_taget, delta_weight, avg_factor, reduction_override)
                # loss_dflz = F.cross_entropy(delta_score, delta_taget, delta_weight, reduction='none')
                # loss_dflz = self.loss_dfl.loss_weight * reduce_loss(
                #     loss_dflz, delta_weight, self.loss_dfl.reduction, avg_factor)
                # print(f'loss_dflz = {loss_dfl} == {loss_dflz} ? {loss_dfl == loss_dflz}')

        return loss_dfl


def dprint(*args, **kwargs):
    import os
    if 'DEBUG' in os.environ:
        print(*args, **kwargs)


def encode_box(xyxy):
    # z=(h*w).sqrt().log2(); r=(h/w).log2()
    # ?? 论文中 r=(h/w).sqrt().log2()
    xy = 0.5 * (xyxy[..., 0:2] + xyxy[..., 2:4])
    wh = xyxy[..., 2:4] - xyxy[..., 0:2]
    xy[xy <= 0] = 0.
    wh[wh <= 0] = 1.
    # if wh[(wh < 0)].numel() > 0:
    #     print(f'\nwh=> <0 numel:{wh[wh < 0].numel()}, >0 numel:{wh[wh > 0].numel()}\n')
    z = (wh).prod(-1, keepdim=True).sqrt().log2()
    r = (wh[..., 1:2] / wh[..., 0:1]).log2()
    # NOTE: xyzr **not** learnable
    xyzr = torch.cat([xy, z, r], dim=-1).detach()
    # if any(xyzr.flatten().isnan()):
    #     print(f'\nencode_box=> NaN numel:{xyzr[xyzr.isnan()].numel()}, >0 numel:{xyzr[xyzr > 0].numel()}\n')
    return xyzr


def decode_box(xyzr):
    scale = 2.00 ** xyzr[..., 2:3]
    ratio = 2.00 ** torch.cat([xyzr[..., 3:4] * -0.5, xyzr[..., 3:4] * 0.5], dim=-1)
    wh = scale * ratio
    xy = xyzr[..., 0:2]
    xyxy = torch.cat([xy - wh * 0.5, xy + wh * 0.5], dim=-1)
    return xyxy


def position_embedding(token_xyzr, num_feats, temperature=10000):
    assert token_xyzr.size(-1) == 4
    term = token_xyzr.new_tensor([1000, 1000, 1, 1]).view(1, 1, -1)
    token_xyzr = token_xyzr / term
    dim_t = torch.arange(num_feats, dtype=torch.float32, device=token_xyzr.device)
    dim_t = (temperature ** (2 * (dim_t // 2) / num_feats)).view(1, 1, 1, -1)
    pos_x = token_xyzr[..., None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=4).flatten(2)
    return pos_x
