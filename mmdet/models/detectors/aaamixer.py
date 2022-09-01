from ..builder import DETECTORS
from .two_stage import TwoStageDetector
from .sparse_rcnn import SparseRCNN
from .base import BaseDetector
from mmcv.runner import auto_fp16

# for increment learning
import cv2, copy, mmcv, torch
import numpy as np
from torch import Tensor
from typing import Any
from collections import OrderedDict
from torchvision.transforms import ToPILImage
from PIL import Image, ImageDraw, ImageFont
from ..builder import build_backbone, build_head, build_neck
from .. import build_detector


@DETECTORS.register_module()
class AaaMixer(BaseDetector):
    '''
    We hack and build our model into Sparse RCNN framework implementation in mmdetection.
    '''
    def __init__(self,
                 backbone=None,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 teacher_config=None,
                 teacher_ckpt=None,
                 teacher_test_cfg=None,
                 eval_teacher=True):
        super(AaaMixer, self).__init__(init_cfg)
        self.has_teacher = teacher_config and teacher_ckpt
        assert len(teacher_test_cfg.rcnn.score_thr) == roi_head.num_stages
        assert len(teacher_test_cfg.rcnn.out_stage) <= roi_head.num_stages
        assert max(teacher_test_cfg.rcnn.out_stage) <= roi_head.num_stages - 1

        # froked from TwoStageDetector
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)
        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            roi_head.update(teacher_test_cfg=teacher_test_cfg.rcnn)
            roi_head.update(has_teacher=self.has_teacher)
            self.roi_head = build_head(roi_head)
        assert self.with_rpn, 'AaaMixer do not support external proposals'

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.teacher_test_cfg = teacher_test_cfg

        # data info
        self.Label2CatNameId = dict()  # {Label: [CatID, CatName], ...}
        self.LableInPCNTask = {'prev':[], 'curr': [], 'next': []}
        # prev：先前已学完的任务Label/ curr: 正在学习的任务Label/ next: 随后再学的类别的Lable

        # # Build teacher model from config file
        # if self.has_teacher:
        #     self.eval_teacher = eval_teacher
        #     self.teacher_model = self.set_teacher(config=teacher_config, ckptfile=teacher_ckpt, trainval='val')
        #     print(f'教师模型加载成功，{teacher_ckpt}')
        # else:
        #     self.teacher_model = None
        #     print(f'教师模型未设置，teacher_config={teacher_config}，teacher_ckpt={teacher_ckpt}')

    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None

    def set_teacher(self, config=None, ckptfile=None, model=None, trainval='val'):
        # Build teacher model by student API
        if (config is None or ckptfile is None) and model is None:
            self.has_teacher = False
            self.roi_head.has_teacher = False
            for i in range(self.roi_head.num_stages):
                self.roi_head.bbox_head[i].has_teacher = False
            print(f'教师模型未设置')
            return None
        if config and ckptfile:
            if isinstance(config, str):
                config = mmcv.Config.fromfile(config)
            self.teacher_model = build_detector(config['model'])
            if ckptfile:
                mmcv.runner.load_checkpoint(self.teacher_model, ckptfile, map_location='cpu')
        elif model is not None:
            self.teacher_model = copy.deepcopy(model)
        else:
            raise NotImplementedError('教师模型无法设置')
        if trainval == 'val':
            self.eval_teacher = True
            # self.teacher_model.eval()
            self.teacher_model.train(False)
            for name, param in self.teacher_model.named_parameters():
                param.requires_grad = False
        else:
            self.eval_teacher = False
            self.teacher_model.train(True)
        # del teacher of teacher
        if getattr(self.teacher_model, 'teacher_model', None) is not None:
            setattr(self.teacher_model, 'teacher_model', None)
        if getattr(self.teacher_model, 'has_teacher', False):
            self.teacher_model.has_teacher = False
            self.teacher_model.roi_head.has_teacher = False
            for i in range(self.teacher_model.roi_head.num_stages):
                self.teacher_model.roi_head.bbox_head[i].has_teacher = False
        # set teacher of student
        self.has_teacher = True
        self.roi_head.has_teacher = True
        for i in range(self.roi_head.num_stages):
            self.roi_head.bbox_head[i].has_teacher = True
        print(f'教师模型已设置，TrainVal：{trainval}，权值加载：{ckptfile if not model else "byModel"}')
        return self.teacher_model

    def out_teacher(self, img, img_metas, rescale=False):
        assert self.has_teacher, '当前没有教师模型'
        with torch.no_grad():
            neck_feat = self.teacher_model.extract_feat(img)
            proposal_boxes, proposal_features, imgs_whwh, semproto, geoproto = \
                self.teacher_model.rpn_head.simple_test_rpn(neck_feat, img_metas)
            head_outs = None
            pred_outs = self.teacher_model.roi_head.complex_test(
                neck_feat,
                proposal_boxes,
                proposal_features,
                img_metas,
                imgs_whwh=imgs_whwh,
                rescale=rescale)
            # print(len(pred_outs))
        return neck_feat, pred_outs

    def set_student(self, ckptfile=None):
        if ckptfile is not None:
            print(f'学生模型权值加载：{ckptfile}')
            mmcv.runner.load_checkpoint(self, ckptfile, map_location='cpu')
        # Frozen
        # Loss
        return self

    def load_student(self, ckptfile):
        mmcv.runner.load_checkpoint(self, ckptfile, map_location='cpu')
        # delete prev teacher of student
        if self.teacher_model is not None:
            self.teacher_model = None
            self.has_teacher = False
        return None

    def set_datainfo(self, cat2id: dict, cat2label: dict, pred_cat=[], load_cat=[], task_cat=[]):
        # cat2id: {CatName: CatID, ...} cat2label: {CatID: Label, ...}
        catid2catname = {v: k for k, v in cat2id.items()}
        self.Label2CatNameId = {v: [catid2catname[k], k] for k, v in cat2label.items()}
        all_cat = []
        for cat in task_cat: all_cat.extend(cat)
        prev_label = [cat2label[cat2id[cat]] for cat in list(set(pred_cat) - set(load_cat))]
        curr_label = [cat2label[cat2id[cat]] for cat in load_cat]
        next_label = [cat2label[cat2id[cat]] for cat in list(set(all_cat) - set(pred_cat))]
        self.LableInPCNTask = {'prev': prev_label, 'curr': curr_label, 'next': next_label}

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """Forward function of SparseR-CNN in train stage.

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (List[Tensor], optional) : Segmentation masks for
                each box. But we don't support it in this architecture.
            proposals (List[Tensor], optional): override rpn proposals with
                custom proposals. Use when `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        assert proposals is None, 'Sparse R-CNN does not support external proposals'
        assert gt_masks is None, 'Sparse R-CNN does not instance segmentation'

        teacher_feat = None
        teacher_info = [None] * self.roi_head.num_stages
        if self.has_teacher:
            teacher_feat, teacher_info = self.out_teacher(img, img_metas, rescale=False)
        assert len(teacher_info) == self.roi_head.num_stages

        # for batch_idx, img_meta in enumerate(img_metas):
        #     stage = self.teacher_test_cfg.rcnn.out_stage[-1]
        #     if teacher_info[stage]['pred_label'][batch_idx].numel() == 0:
        #         continue
        #     target = {'labels': teacher_info[stage]['pred_label'][batch_idx],
        #               'scores': teacher_info[stage]['cls_score'][batch_idx].max(1)[0],
        #               'boxes': teacher_info[stage]['pred_bbox'][batch_idx]}
        #     self.draw_boxes_on_img_v1(
        #         img_info=img_meta, target=target, target_style='style1',
        #         coord='x1y1x2y2', isnorm=False, imgsize='new',
        #         waitKey=-200, window='imgshow', realtodo=1)
        #     # print(target['labels'])

        x = self.extract_feat(img)
        proposal_boxes, proposal_features, imgs_whwh, semproto, geoproto = self.rpn_head.forward_train(x, img_metas)
        roi_losses = self.roi_head.forward_train(
            x,
            proposal_boxes,
            proposal_features,
            img_metas,
            gt_labels,
            gt_bboxes,
            gt_bboxes_ignore=gt_bboxes_ignore,
            gt_masks=gt_masks,
            imgs_whwh=imgs_whwh,
            semproto=semproto,
            geoproto=geoproto,
            teacher_feat=teacher_feat,
            teacher_info=teacher_info,
            task_labels=self.LableInPCNTask
        )
        return roi_losses

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test time augmentation.
        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool): Whether to rescale the results.
                Defaults to False.
        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        x = self.extract_feat(img)
        proposal_boxes, proposal_features, imgs_whwh, semproto, geoproto = self.rpn_head.simple_test_rpn(x, img_metas)
        bbox_results = self.roi_head.simple_test(
            x,
            proposal_boxes,
            proposal_features,
            img_metas,
            imgs_whwh=imgs_whwh,
            rescale=rescale)
        return bbox_results

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        # backbone
        x = self.extract_feat(img)
        # rpn
        num_imgs = len(img)
        dummy_img_metas = [
            dict(img_shape=(800, 1333, 3)) for _ in range(num_imgs)
        ]
        proposal_boxes, proposal_features, imgs_whwh, semproto, geoproto = \
            self.rpn_head.simple_test_rpn(x, dummy_img_metas)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposal_boxes,
                                               proposal_features,
                                               dummy_img_metas,
                                               semproto, geoproto)
        return roi_outs

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        x = self.extract_feats(imgs)
        proposal_list = self.rpn_head.aug_test_rpn(x, img_metas)
        return self.roi_head.aug_test(
            x, proposal_list, img_metas, rescale=rescale)

    def cuda(self, device=None):
        """Since teacher_model is registered as a plain object, it is necessary
        to put the teacher model to cuda when calling cuda function."""
        if self.has_teacher:
            # print('设置教师模型device==>', device)
            self.teacher_model.cuda(device=device)
        return super().cuda(device=device)

    def train(self, mode=True):
        """Set the same train mode for teacher and student model."""
        if self.has_teacher:
            # print(f'设置教师模型训练验证状态 Eval: {self.eval_teacher}')
            if self.eval_teacher:
                self.teacher_model.train(False)
            else:
                self.teacher_model.train(mode)
        super().train(mode)

    def __setattr__(self, name, value):
        """Set attribute, i.e. self.name = value

        This reloading prevent the teacher model from being registered as a
        nn.Module. The teacher module is registered as a plain object, so that
        the teacher parameters will not show up when calling
        ``self.parameters``, ``self.modules``, ``self.children`` methods.
        """
        if name == 'teacher_model' and self.has_teacher:
            object.__setattr__(self, name, value)
        else:
            super().__setattr__(name, value)

    def draw_boxes_on_img_v1(self, img_id=None, img_mat=None, img_info=None,
                             target=Any, target_style='style1',
                             coord='x1y1wh', isnorm=False, imgsize='orig|new',
                             waitKey=200, window='imgshow', realtodo=1):
        # imgsize: 使用原图尺寸或转换后尺寸画图,跟模型中rescale参数协同设定。
        if not realtodo: return
        assert coord in 'x1y1wh|cxcywh|x1y1x2y2'
        img_flip = False
        h_org, w_org, h_new, w_new, w_now, h_now = 0, 0, 0, 0, 0, 0

        print(f'\n加载 Image........')
        if img_id:
            image = self.coco.load_imgs(ids=[img_id])
            target = self.coco.load_anns(ids=[img_id])
        elif img_mat is not None:
            if isinstance(img_mat, Tensor):
                img_mat = ToPILImage()(img_mat)
            image = img_mat
        elif img_info:
            if isinstance(img_info, dict) and 'filename' in img_info:
                # print('img_info=>', img_info)
                img_path = img_info.get('filename', 'error filename')
                img_flip = img_info.get('flip', False)
                h_org, w_org = img_info.get('orig_size', img_info.get('ori_shape', [None] * 3)[:2])
                h_new, w_new = img_info.get('size', img_info.get('img_shape', [None] * 3)[:2])
            else:
                img_path = img_info
            image = Image.open(img_path)
        else:
            raise ValueError('无法加载图片')
        image = image.convert('RGB')
        if imgsize == 'new':
            image = image.resize((w_new, h_new), Image.ANTIALIAS)
        w_now, h_now = image.size
        print(f'图像尺寸信息: [h_org, w_org], [h_new, w_new], [h_now, w_now]'
              f'= {h_org, w_org, h_new, w_new, h_now, w_now}')
        # image.show()

        print(f'加载 Target........')
        if target_style == 'style1':
            # boxes, labels, scores 按找字典传入
            boxes = target['boxes']
            labels = target.get('labels', [0] * len(boxes))
            scores = target.get('scores', [0] * len(boxes))
            boxes = boxes if not isinstance(boxes, Tensor) else boxes.cpu().numpy().tolist()
            labels = labels if not isinstance(labels, Tensor) else labels.cpu().numpy().tolist()
            scores = scores if not isinstance(scores, Tensor) else scores.cpu().numpy().tolist()
            assert len(labels) == len(boxes) and len(scores) == len(boxes)
            target = list(zip(labels, scores, boxes))
        elif target_style == 'mmpred':
            # 直接传入mmde中 target=model.get_bboxes() 的预测输出: [(x1, y1, x2, y2, score), ...][label, ...]
            # [(x1, y1, x2, y2, score, label), ...]=>[((x1, y1, x2, y2), score, label), ...]
            target = [torch.cat([t[0], t[1].unsqueeze(1)], dim=1) for t in target]
            if isinstance(target[0], Tensor):
                target = [t.cpu().numpy().tolist() for t in target]
            target = [[t[5], t[4], t[:4]] for t in target]
        else:
            raise NotImplementedError(f'错误的Taget存放方式, target_style={target_style}')

        print(f'绘制 BBOX........')
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationMono-Italic.ttf", 18)
        for idx, (label, score, bbox) in enumerate(target):
            # print(label, score, bbox)
            if coord == 'x1y1wh':
                x1, y1, w, h = (int(v) for v in bbox)
                x_min, y_min, x_max, y_max = x1, y1, x1 + w, y1 + h
            elif coord == 'cxcywh':
                cx, cy, w, h = (int(v) for v in bbox)
                x_min, y_min, x_max, y_max = int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2)
            elif coord == 'x1y1x2y2':
                x_min, y_min, x_max, y_max = (int(v) for v in bbox)
            else:
                raise NotImplementedError(f'参数错误：coord={coord}')
            if img_flip:
                x_min, y_min, x_max, y_max = w_now - x_max, y_min, w_now - x_min, y_max
            draw.line([(x_min, y_min), (x_min, y_max), (x_max, y_max),
                       (x_max, y_min), (x_min, y_min)], width=1, fill=(0, 0, 255))
            # CategoryName, CategoryID, CategoryLabel
            text = self.Label2CatNameId[label][0] + ['', '|' + str(score)[:5]][score > 0]
            draw.text((x_min, y_min), text, (255, 255, 0), font=font)
        # image.show()
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        cv2.imshow(f'{window}', image)
        print(f'绘制完成........')
        cv2.waitKey(waitKey)

