import torch
import torch.nn as nn
from mmcv.runner import BaseModule

from mmdet.models.builder import HEADS
from ...core import bbox_cxcywh_to_xyxy


@HEADS.register_module()
class InitialQueryGenerator(BaseModule):
    """
    This module produces initial content vector $\mathbf{q}$ and positional vector $(x, y, z, r)$.
    Note that the initial positional vector is **not** learnable.
    """
    def __init__(self,
                 num_query=100,
                 content_dim=256,
                 init_cfg=None,
                 method='xyzr',
                 num_classes=80,
                 **kwargs):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
                                 'behavior, init_cfg is not allowed to be set'
        assert method in ['xyzr', 'xyxy', 'xyxy+semproto', 'xyxy+geoproto''xyxy+semproto+geoproto']
        super(InitialQueryGenerator, self).__init__(init_cfg)
        self.num_query = num_query
        self.content_dim = content_dim
        self.method = method
        self.num_classes = num_classes
        self._init_layers()

    def _init_layers(self):
        self.init_proposal_bboxes = nn.Embedding(self.num_query, 4)
        self.init_content_features = nn.Embedding(self.num_query, self.content_dim)
        if 'semproto' in self.method:   # 语义概念原型先验
            self.init_semantic_prototype = nn.Embedding(self.num_classes, self.content_dim)
        if 'geoproto' in self.method:   # 几何形状原型先验
            self.init_geometry_prototype = nn.Embedding(self.num_classes, self.content_dim)

    def init_weights(self):
        super(InitialQueryGenerator, self).init_weights()
        nn.init.constant_(self.init_proposal_bboxes.weight[:, :2], 0.5)
        nn.init.constant_(self.init_proposal_bboxes.weight[:, 2:], 1)
        if 'semproto' in self.method:
            nn.init.constant_(self.init_semantic_prototype.weight, 0)       # 语艺原型特征初始化为0
        if 'geoproto' in self.method:
            nn.init.constant_(self.init_semantic_prototype.weight, 0)       # 原型特征初始化为0

    def _ccwh_into_xyzr_feat(self, imgs, img_metas):
        """
        Hacks based on 'sparse_roi_head.py'.
        For the positional vector, we first compute (x, y, z, r) that fully covers an image.
        """
        proposals = self.init_proposal_bboxes.weight.clone()
        proposals = bbox_cxcywh_to_xyxy(proposals)
        num_imgs = len(imgs[0])
        imgs_whwh = []
        for meta in img_metas:
            h, w, _ = meta['img_shape']
            imgs_whwh.append(imgs[0].new_tensor([[w, h, w, h]]))
        imgs_whwh = torch.cat(imgs_whwh, dim=0)
        imgs_whwh = imgs_whwh[:, None, :]

        proposals = proposals * imgs_whwh

        # z=(h*w).sqrt().log2(); r=(h/w).log2()  ?? 论文中 r=(h/w).sqrt().log2()
        xy = 0.5 * (proposals[..., 0:2] + proposals[..., 2:4])
        wh = proposals[..., 2:4] - proposals[..., 0:2]
        z = (wh).prod(-1, keepdim=True).sqrt().log2()
        r = (wh[..., 1:2]/wh[..., 0:1]).log2()

        # NOTE: xyzr **not** learnable
        xyzr = torch.cat([xy, z, r], dim=-1).detach()

        feature = self.init_content_features.weight.clone()
        feature = feature[None].expand(num_imgs, *feature.size())
        feature = torch.layer_norm(feature, normalized_shape=[feature.size(-1)])

        return xyzr, feature, imgs_whwh, None, None

    def _ccwh_into_xyxy_feat(self, imgs, img_metas):
        proposals = self.init_proposal_bboxes.weight.clone()
        proposals = bbox_cxcywh_to_xyxy(proposals)
        num_imgs = len(imgs[0])
        imgs_whwh = []
        for meta in img_metas:
            h, w, _ = meta['img_shape']
            imgs_whwh.append(imgs[0].new_tensor([[w, h, w, h]]))
        imgs_whwh = torch.cat(imgs_whwh, dim=0)
        imgs_whwh = imgs_whwh[:, None, :]
        xyxy = (proposals * imgs_whwh).detach()    # (N, numQ, 4)

        feature = self.init_content_features.weight.clone()
        feature = feature[None].expand(num_imgs, *feature.size())
        feature = torch.layer_norm(feature, normalized_shape=[feature.size(-1)])

        return xyxy, feature, imgs_whwh, None, None

    def _ccwh_into_xyxy_proto_feat(self, imgs, img_metas):
        proposals = self.init_proposal_bboxes.weight.clone()
        proposals = bbox_cxcywh_to_xyxy(proposals)
        num_imgs = len(imgs[0])
        imgs_whwh = []
        for meta in img_metas:
            h, w, _ = meta['img_shape']
            imgs_whwh.append(imgs[0].new_tensor([[w, h, w, h]]))
        imgs_whwh = torch.cat(imgs_whwh, dim=0)
        imgs_whwh = imgs_whwh[:, None, :]
        xyxy = (proposals * imgs_whwh).detach()    # (N, numQ, 4)

        feature = self.init_content_features.weight.clone()
        feature = feature[None].expand(num_imgs, *feature.size())
        feature = torch.layer_norm(feature, normalized_shape=[feature.size(-1)])

        semproto, geoproto = None, None
        if 'semproto' in self.method:
            semproto = self.init_semantic_prototype.weight.clone()
            semproto = torch.layer_norm(semproto, normalized_shape=[semproto.size(-1)])
        if 'geoproto' in self.method:
            geoproto = self.init_geometry_prototype.weight.clone()
            geoproto = torch.layer_norm(geoproto, normalized_shape=[geoproto.size(-1)])

        return xyxy, feature, imgs_whwh, semproto, geoproto

    def _decode_init_proposals(self, imgs, img_metas):
        if self.method == 'xyzr':
            return self._ccwh_into_xyzr_feat(imgs, img_metas)
        elif self.method == 'xyxy':
            return self._ccwh_into_xyxy_feat(imgs, img_metas)
        elif self.method in ['xyxy+semproto', 'xyxy+geoproto', 'xyxy+semproto+geoproto']:
            return self._ccwh_into_xyxy_proto_feat(imgs, img_metas)
        else:
            raise NotImplementedError(f'未知的编码方法 self.method={self.method}')

    def forward_dummy(self, img, img_metas):
        """Dummy forward function.
        Used in flops calculation.
        """
        return self._decode_init_proposals(img, img_metas)

    def forward_train(self, img, img_metas):
        """Forward function in training stage."""
        return self._decode_init_proposals(img, img_metas)

    def simple_test_rpn(self, img, img_metas):
        """Forward function in testing stage."""
        return self._decode_init_proposals(img, img_metas)
