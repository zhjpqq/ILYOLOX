import contextlib
import itertools
import logging
import os.path as osp
import tempfile
import warnings
from collections import OrderedDict

import mmcv
import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable

from mmdet.core import eval_recalls
from .api_wrappers import COCO, COCOeval
from .builder import DATASETS
from .custom import CustomDataset

import io
import cv2
import random
from typing import Any
from torch import Tensor
from torchvision.transforms import ToPILImage
from PIL import Image, ImageDraw, ImageFont


@DATASETS.register_module()
class WRXTDataset(CustomDataset):

    ALL_CLASSES_IDS = {'Fighter': 0, 'Helicopter': 1, 'Plane': 2, 'Aircraft': 3,
                       'Tank': 4, 'Armored': 5, 'SUV': 6, 'Launcher': 7, 'Solider': 8}

    CLASSES = ('Fighter', 'Helicopter', 'Plane', 'Aircraft', 'Tank', 'Armored', 'SUV', 'Launcher', 'Solider')
    CLASSES_ENCN = {'Fighter': '战斗机', 'Helicopter': '直升机', 'Plane': '民航机', 'Aircraft': '航天飞机',
                    'Tank': '坦克', 'Armored': '装甲车', 'SUV': '越野车', 'Launcher': '导弹发射车', 'Solider': '士兵'}

    # K6U3
    # CLASSES = ('Fighter', 'Helicopter', 'Plane', 'Aircraft', 'Armored', 'Solider')
    # CLASSES_ENCN = {'Fighter': '战斗机', 'Helicopter': '直升机', 'Plane': '民航机', 'Aircraft': '航天飞机',
    #                 'Tank': '坦克', 'Armored': '装甲车', 'SUV': '越野车', 'Launcher': '导弹发射车', 'Solider': '士兵'}

    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.
        Args:
            ann_file (str): Path of annotation file.
        Returns:
            list[dict]: Annotation info from COCO api.
        """
        self.coco = COCO(ann_file)
        # The order of returned `cat_ids` will not change with the order of the ALL_CLASSES
        self.cat_ids_pred = self.coco.get_cat_ids(cat_names=self.PRED_CLASSES)
        self.cat_ids_load = self.coco.get_cat_ids(cat_names=self.LOAD_CLASSES)
        self.cat_ids_all = self.coco.get_cat_ids(cat_names=self.ALL_CLASSES)
        # 按All_Classes分配全局Label，确保OneHot与模型输出类别数相同
        self.cat2label = {cat_id: i + self.START_LABEL for i, cat_id in enumerate(self.cat_ids_all)}
        self.label2cat = {lable: catid for (catid, lable) in self.cat2label.items()}
        self.img_ids_load = self.coco.get_img_ids(cat_ids=[self.ALL_CLASSES_IDS[k] for k in self.LOAD_CLASSES])
        if self.imgpercent < 1:
            # TODO，样本量较小时，个别类没有样本，重写确保每个加载类都有样本被选中
            self.img_ids_load = random.sample(self.img_ids_load, int(len(self.img_ids_load)*self.imgpercent))
            print(f'DataIL: 随机筛选数据集，筛选率:imgpercent={self.imgpercent},筛选后图像{len(self.img_ids_load)}张')
        print(f'DataIL: 当前加载图像数量为：{len(self.img_ids_load)} 张！')
        # self.img_ids_all = self.coco.get_img_ids(cat_ids=[])
        # self.img_ids_other = set(self.img_ids_all) - set(self.img_ids_load)
        # self.img_ids_other = [idx for idx in self.img_ids_all if idx not in self.img_ids_load]
        data_infos = []
        total_ann_ids = []
        for i in self.img_ids_load:
            info = self.coco.load_imgs([i])[0]
            info['filename'] = info['file_name']
            data_infos.append(info)
            ann_ids = self.coco.get_ann_ids(img_ids=[i], cat_ids=self.cat_ids_load)  # TODO ? no cat_ids
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
        return data_infos

    def get_ann_info(self, idx):
        """Get COCO annotation by index.
        Args: abcdefg 增加类别过滤(cat_ids)
            idx (int): Index of data.
        Returns:
            dict: Annotation info of specified index.
        """
        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id], cat_ids=self.cat_ids_load)
        ann_info = self.coco.load_anns(ann_ids)
        # filter = [ann['category_id'] in self.cat_ids_load for ann in ann_info]
        # assert all(filter), f'筛选出现错误类别:{print(img_id, filter)}'
        return self._parse_ann_info(self.data_infos[idx], ann_info)

    def get_cat_ids(self, idx):
        """Get COCO category ids by index.
        Args:
            idx (int): Index of data.
        Returns:
            list[int]: All categories in the image of specified index.
        """
        raise ValueError('get_cat_ids 未使用过？')
        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        # ann_ids = self.coco.get_ann_ids(img_ids=[img_id], cat_ids=self.cat_ids_load)
        ann_info = self.coco.load_anns(ann_ids)
        return [ann['category_id'] for ann in ann_info]

    def get_cat_names(self, idx):
        """通过IDX获取类别名称"""
        ALL_IDS_CLASSES = {v: k for k, v in self.ALL_CLASSES_IDS.items()}
        if isinstance(idx, int):
            idx = [idx]
        classes = [ALL_IDS_CLASSES[x] for x in idx]
        return classes

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        # obtain images that contain annotation
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        # obtain images that contain annotations of the required categories
        ids_in_cat = set()
        for i, class_id in enumerate(self.cat_ids_pred):
            ids_in_cat |= set(self.coco.cat_img_map[class_id])
        # merge the image id sets of the two conditions and use the merged set
        # to filter out images if self.filter_empty_gt=True
        ids_in_cat &= ids_with_ann

        valid_img_ids = []
        for i, img_info in enumerate(self.data_infos):
            img_id = self.img_ids_load[i]
            if self.filter_empty_gt and img_id not in ids_in_cat:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
                valid_img_ids.append(img_id)
        self.img_ids_load = valid_img_ids
        return valid_inds

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.
        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.
        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids_pred:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann.get('segmentation', None))

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann

    def xyxy2xywh(self, bbox):
        """Convert ``xyxy`` style bounding boxes to ``xywh`` style for COCO
        evaluation.
        Args:
            bbox (numpy.ndarray): The bounding boxes, shape (4, ), in
                ``xyxy`` order.
        Returns:
            list[float]: The converted bounding boxes, in ``xywh`` order.
        """

        _bbox = bbox.tolist()
        return [
            _bbox[0],
            _bbox[1],
            _bbox[2] - _bbox[0],
            _bbox[3] - _bbox[1],
        ]

    def _proposal2json(self, results):
        """Convert proposal results to COCO json style."""
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids_load[idx]
            bboxes = results[idx]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = self.xyxy2xywh(bboxes[i])
                data['score'] = float(bboxes[i][4])
                data['category_id'] = 1
                json_results.append(data)
        return json_results

    def _det2json(self, results):
        """Convert detection results to COCO json style."""
        json_results = []
        # 循环图像数量，eg val 5K张
        for idx in range(len(self)):
            img_id = self.img_ids_load[idx]
            result = results[idx]
            # 循环类别数量，模型输出的类别预测数量，不是数据集中的类别数
            for label in range(len(result)):
                # print('result===> ', result)
                bboxes = result[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    # data['category_id'] = self.cat_ids_pred[label]
                    data['category_id'] = self.label2cat[label]
                    json_results.append(data)
        return json_results

    def _segm2json(self, results):
        """Convert instance segmentation results to COCO json style."""
        bbox_json_results = []
        segm_json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids_load[idx]
            det, seg = results[idx]
            for label in range(len(det)):
                # bbox results
                bboxes = det[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = self.cat_ids_pred[label]
                    bbox_json_results.append(data)

                # segm results
                # some detectors use different scores for bbox and mask
                if isinstance(seg, tuple):
                    segms = seg[0][label]
                    mask_score = seg[1][label]
                else:
                    segms = seg[label]
                    mask_score = [bbox[4] for bbox in bboxes]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(mask_score[i])
                    data['category_id'] = self.cat_ids_pred[label]
                    if isinstance(segms[i]['counts'], bytes):
                        segms[i]['counts'] = segms[i]['counts'].decode()
                    data['segmentation'] = segms[i]
                    segm_json_results.append(data)
        return bbox_json_results, segm_json_results

    def results2json(self, results, outfile_prefix):
        """Dump the detection results to a COCO style json file.

        There are 3 types of results: proposals, bbox predictions, mask
        predictions, and they have different data types. This method will
        automatically recognize the type, and dump them to json files.

        Args:
            results (list[list | tuple | ndarray]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
                "somepath/xxx.proposal.json".

        Returns:
            dict[str: str]: Possible keys are "bbox", "segm", "proposal", and \
                values are corresponding filenames.
        """
        result_files = dict()
        if isinstance(results[0], list):
            json_results = self._det2json(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            mmcv.dump(json_results, result_files['bbox'])
        elif isinstance(results[0], tuple):
            json_results = self._segm2json(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            result_files['segm'] = f'{outfile_prefix}.segm.json'
            mmcv.dump(json_results[0], result_files['bbox'])
            mmcv.dump(json_results[1], result_files['segm'])
        elif isinstance(results[0], np.ndarray):
            json_results = self._proposal2json(results)
            result_files['proposal'] = f'{outfile_prefix}.proposal.json'
            mmcv.dump(json_results, result_files['proposal'])
        else:
            raise TypeError('invalid type of results')
        return result_files

    def fast_eval_recall(self, results, proposal_nums, iou_thrs, logger=None):
        gt_bboxes = []
        for i in range(len(self.img_ids_load)):
            ann_ids = self.coco.get_ann_ids(img_ids=self.img_ids_load[i])
            ann_info = self.coco.load_anns(ann_ids)
            if len(ann_info) == 0:
                gt_bboxes.append(np.zeros((0, 4)))
                continue
            bboxes = []
            for ann in ann_info:
                if ann.get('ignore', False) or ann['iscrowd']:
                    continue
                x1, y1, w, h = ann['bbox']
                bboxes.append([x1, y1, x1 + w, y1 + h])
            bboxes = np.array(bboxes, dtype=np.float32)
            if bboxes.shape[0] == 0:
                bboxes = np.zeros((0, 4))
            gt_bboxes.append(bboxes)

        recalls = eval_recalls(
            gt_bboxes, results, proposal_nums, iou_thrs, logger=logger)
        ar = recalls.mean(axis=1)
        return ar

    def format_results(self, results, jsonfile_prefix=None, **kwargs):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[tuple | numpy.ndarray]): Testing results of the dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing \
                the json filepaths, tmp_dir is the temporal directory created \
                for saving json files when jsonfile_prefix is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
                format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        result_files = self.results2json(results, jsonfile_prefix)
        return result_files, tmp_dir

    def evaluate_det_segm(self,
                          results,
                          result_files,
                          coco_gt,
                          metrics,
                          logger=None,
                          classwise=False,
                          proposal_nums=(100, 300, 1000),
                          iou_thrs=None,
                          metric_items=None):
        """Instance segmentation and object detection evaluation in COCO
        protocol.

        Args:
            results (list[list | tuple | dict]): Testing results of the
                dataset.
            result_files (dict[str, str]): a dict contains json file path.
            coco_gt (COCO): COCO API object with ground truth annotation.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float], optional): IoU threshold used for
                evaluating recalls/mAPs. If set to a list, the average of all
                IoUs will also be computed. If not specified, [0.50, 0.55,
                0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
                Default: None.
            metric_items (list[str] | str, optional): Metric items that will
                be returned. If not specified, ``['AR@100', 'AR@300',
                'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
                used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
                'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
                ``metric=='bbox' or metric=='segm'``.

        Returns:
            dict[str, float]: COCO style evaluation metric.
        """
        if iou_thrs is None:
            iou_thrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        if metric_items is not None:
            if not isinstance(metric_items, list):
                metric_items = [metric_items]

        eval_results = OrderedDict()
        results_per_category = []
        for metric in metrics:
            msg = f'Evaluating {metric}...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            if metric == 'proposal_fast':
                if isinstance(results[0], tuple):
                    raise KeyError('proposal_fast is not supported for '
                                   'instance segmentation result.')
                ar = self.fast_eval_recall(results, proposal_nums, iou_thrs, logger='silent')
                log_msg = []
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
                    log_msg.append(f'\nAR@{num}\t{ar[i]:.4f}')
                log_msg = ''.join(log_msg)
                print_log(log_msg, logger=logger)
                continue

            iou_type = 'bbox' if metric == 'proposal' else metric
            if metric not in result_files:
                raise KeyError(f'{metric} is not in results')
            try:
                predictions = mmcv.load(result_files[metric])
                if iou_type == 'segm':
                    # Refer to https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py#L331  # noqa
                    # When evaluating mask AP, if the results contain bbox,
                    # cocoapi will use the box area instead of the mask area
                    # for calculating the instance area. Though the overall AP
                    # is not affected, this leads to different
                    # small/medium/large mask AP results.
                    for x in predictions:
                        x.pop('bbox')
                    warnings.simplefilter('once')
                    warnings.warn(
                        'The key "bbox" is deleted for more accurate mask AP '
                        'of small/medium/large instances since v2.12.0. This '
                        'does not change the overall mAP calculation.',
                        UserWarning)
                coco_det = coco_gt.loadRes(predictions)
            except IndexError:
                print_log(
                    'The testing results of the whole dataset is empty.',
                    logger=logger,
                    level=logging.ERROR)
                break

            cocoEval = COCOeval(coco_gt, coco_det, iou_type)
            cocoEval.params.catIds = self.cat_ids_pred
            cocoEval.params.imgIds = self.img_ids_load
            cocoEval.params.maxDets = list(proposal_nums)
            cocoEval.params.iouThrs = iou_thrs
            # mapping of cocoEval.stats
            coco_metric_names = {
                'mAP': 0,
                'mAP_50': 1,
                'mAP_75': 2,
                'mAP_s': 3,
                'mAP_m': 4,
                'mAP_l': 5,
                'AR@100': 6,
                'AR@300': 7,
                'AR@1000': 8,
                'AR_s@1000': 9,
                'AR_m@1000': 10,
                'AR_l@1000': 11
            }
            if metric_items is not None:
                for metric_item in metric_items:
                    if metric_item not in coco_metric_names:
                        raise KeyError(f'metric item {metric_item} is not supported')

            if metric == 'proposal':
                cocoEval.params.useCats = 0
                cocoEval.evaluate()
                cocoEval.accumulate()

                # Save coco summarize print information to logger
                redirect_string = io.StringIO()
                with contextlib.redirect_stdout(redirect_string):
                    cocoEval.summarize()
                print_log('\n' + redirect_string.getvalue(), logger=logger)

                if metric_items is None:
                    metric_items = [
                        'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000',
                        'AR_m@1000', 'AR_l@1000'
                    ]

                for item in metric_items:
                    val = float(f'{cocoEval.stats[coco_metric_names[item]]:.3f}')
                    eval_results[item] = val
            else:
                cocoEval.evaluate()
                cocoEval.accumulate()

                # Save coco summarize print information to logger
                redirect_string = io.StringIO()
                with contextlib.redirect_stdout(redirect_string):
                    cocoEval.summarize()
                print_log('\n' + redirect_string.getvalue(), logger=logger)

                if classwise or self.catwise:  # Compute per-category AP
                    # Compute per-category AP
                    # from https://github.com/facebookresearch/detectron2/
                    precisions = cocoEval.eval['precision']
                    # precision: (iou, recall, cls, area range, max dets)
                    print(f'precisions.shape = {precisions.shape} ？= {len(self.cat_ids_pred)}')
                    assert len(self.cat_ids_pred) == precisions.shape[2], \
                        print(len(self.cat_ids_pred), ' ≠ ', precisions.shape[2])
                    # TODO 1,去除掉precisions中模型预测的FutuClass，只保留NowClass
                    # TODO 2,补全cat_ids_pred=cat_ids_all，FutuClass虽计算但不用

                    for idx, catId in enumerate(self.cat_ids_pred):
                        # area range index 0: all area ranges
                        # max dets index -1: typically 100 per image
                        nm = self.coco.loadCats(catId)[0]
                        precision = precisions[:, :, idx, 0, -1]
                        precision = precision[precision > -1]
                        if precision.size:
                            ap = np.mean(precision)
                        else:
                            ap = float('nan')
                        results_per_category.append((f'{nm["name"]}', f'{float(ap):0.3f}'))
                    num_columns = min(6, len(results_per_category) * 2)
                    results_flatten = list(itertools.chain(*results_per_category))
                    headers = ['category', 'AP'] * (num_columns // 2)
                    results_2d = itertools.zip_longest(*[
                        results_flatten[i::num_columns]
                        for i in range(num_columns)
                    ])
                    table_data = [headers]
                    table_data += [result for result in results_2d]
                    table = AsciiTable(table_data)
                    print_log(f'\n=============== CategoryWise,各个类别测试结果 预测{len(self.PRED_CLASSES)}类 '
                              f'加载{len(self.LOAD_CLASSES)}类============', logger=logger)
                    print_log(table.table, logger=logger)
                    results_per_category = OrderedDict([(k, float(v)) for k, v in results_per_category])

                if metric_items is None:
                    metric_items = ['mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l']

                for metric_item in metric_items:
                    key = f'{metric}_{metric_item}'
                    val = float(f'{cocoEval.stats[coco_metric_names[metric_item]]:.3f}')
                    eval_results[key] = val
                ap = cocoEval.stats[:6]
                eval_results[f'{metric}_mAP_copypaste'] = (
                    f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                    f'{ap[4]:.3f} {ap[5]:.3f}')
        if not isinstance(results_per_category, OrderedDict):
            results_per_category = OrderedDict(results_per_category)
        return eval_results, results_per_category

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=None,
                 metric_items=None):
        """Evaluation in COCO protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
                # [imgNums, classNums, objNums, 4+1]
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float], optional): IoU threshold used for
                evaluating recalls/mAPs. If set to a list, the average of all
                IoUs will also be computed. If not specified, [0.50, 0.55,
                0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
                Default: None.
            metric_items (list[str] | str, optional): Metric items that will
                be returned. If not specified, ``['AR@100', 'AR@300',
                'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
                used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
                'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
                ``metric=='bbox' or metric=='segm'``.
            catsplit: 增量式类别划分，[('', ''), ('', ''), ('', ''), ...]
        Returns:
            dict[str, float]: COCO style evaluation metric.
        """
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        coco_gt = self.coco
        # self.cat_ids_pred = coco_gt.get_cat_ids(cat_names=self.ALL_CLASSES)

        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)
        eval_results, results_per_category = self.evaluate_det_segm(
            results, result_files, coco_gt, metrics, logger,
            classwise, proposal_nums, iou_thrs, metric_items)

        # 统计当前各任务阶段测试结果mAP
        task_mAP_list = []
        cats_map_dict = {}
        task_mAP_dict = {f'Task{i + 1}({len(task_cat)}类)': 0 for i, task_cat in enumerate(self.TASK_CLASSES)}
        for tid, task_cat in enumerate(self.TASK_CLASSES):
            res = {k: results_per_category[k] for k in task_cat if k in results_per_category.keys()}
            task_mAP_list.append(res)
            cats_map_dict.update(res)
            avg = sum(res.values()) / len(res.values()) if res else 0
            task_mAP_dict[f'Task{tid + 1}({len(task_cat)}类)'] = round(avg, 5)
        task_mAP_dict.update({f'TaskAvgMAP': round(sum(cats_map_dict.values()) / len(cats_map_dict), 5)})
        print_log(f'\n========= 各任务阶段测试结果 =========', logger=logger)
        print_log(f'task_mAP_list==> ' + str(task_mAP_list), logger=logger)
        print_log(f'task_mAP_dict==> ' + str(task_mAP_dict), logger=logger)
        print_log(f'==================================\n', logger=logger)

        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results

    def draw_boxes_on_img_v1(self, img_id=None, img_mat=None, img_info=None, labels=None, boxes=None,
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
        elif img_mat:
            if isinstance(img_mat, Tensor):
                img_mat = ToPILImage()(img_mat)
            image = img_mat
        elif img_info:
            if isinstance(img_info, dict) and 'filename' in img_info:
                print('img_info=>', img_info)
                img_path = img_info.get('filename', None)
                img_flip = img_info.get('flip', False)
                h_org, w_org = img_info.get('orig_size', img_info.get('ori_shape', [None]*3)[:2])
                h_new, w_new = img_info.get('size', img_info.get('img_shape', [None]*3)[:2])
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
        assert len(labels) == len(boxes)
        target = list(zip(labels, boxes))

        print(f'绘制 BBOX........')
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationMono-Italic.ttf", 18)
        for idx, (label, bbox) in enumerate(target):
            if coord == 'x1y1wh':
                x1, y1, w, h = (int(v) for v in bbox)
                x_min, y_min, x_max, y_max = x1, y1, x1 + w, y1 + h
            elif coord == 'cxcywh':
                cx, cy, w, h = (int(v) for v in bbox)
                x_min, y_min, x_max, y_max = int(cx-w/2), int(cy-h/2), int(cx+w/2), int(cy+h/2)
            elif coord == 'x1y1x2y2':
                x_min, y_min, x_max, y_max = (int(v) for v in bbox)
            else:
                raise NotImplementedError(f'参数错误：coord={coord}')
            if img_flip:
                x_min, y_min, x_max, y_max = w_now-x_max, y_min, w_now-x_min, y_max
            draw.line([(x_min, y_min), (x_min, y_max), (x_max, y_max),
                       (x_max, y_min), (x_min, y_min)], width=1, fill=(0, 0, 255))
            text = self.ALL_IDS_CLASSES[self.label2cat[label]] + f',{self.label2cat[label]}' + f'|{label}'
            draw.text((x_min, y_min), text, (255, 255, 0), font=font)
        # image.show()
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        cv2.imshow(f'{window}', image)
        print(f'绘制完成........')
        cv2.waitKey(waitKey)
        return

    def draw_boxes_on_img2(self, img_id=None, img_mat=None, img_path=None, target=Any,
                          isxywh=True, isnorm=False, waitKey=200, window='imgshow', realtodo=1):
        if not realtodo: return

        print(f'\n加载 Image........')
        if img_id:
            image = self._load_image(img_id)
            target = self._load_target(img_id)
            image_path = self._image_path(img_id)
        elif (isinstance(img_mat, Tensor) or img_mat) and target:
            if isinstance(img_mat, Tensor):
                img_mat = ToPILImage()(img_mat)
            image = img_mat
        elif img_path and target:
            image = Image.open(img_path)
        else:
            raise ValueError('无法加载图片')
        image = image.convert('RGB')
        w_now, h_now = image.size
        print('image.size: [w_now, h_now]', image.size)
        # image.show()

        print(f'加载 Target........')
        if img_id is not None:
            target = [(t['category_id'], t['bbox']) for t in target]
        elif isinstance(target, list):
            # [{ann1}, {ann2}, ...]
            target = [(t['category_id'], t['bbox']) for t in target]
        elif isinstance(target, dict):
            # image_id, orig_size = target['image_id'], target['orig_size']
            labels, boxes = list(target['labels'].numpy()), list(target['boxes'].numpy().tolist())
            h_org, w_org = target['orig_size'].numpy()
            h_new, w_new = target['size'].numpy()
            print(f'h_org, w_org: {h_org}, {w_org}',
                  f'h_new, w_new = > {h_new}, {w_new}',
                  f'h_now, w_now = > {h_now}, {w_now}', )
            if isnorm:
                boxes = [[box[0]*w_new, box[1]*h_new, box[2]*w_new, box[3]*h_new] for box in boxes]
            target = list(zip(labels, boxes))

        print(f'绘制 BBOX........')
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationMono-Italic.ttf", 18)
        for idx, (label, bbox) in enumerate(target):
            if isxywh:
                x1, y1, w, h = (int(v) for v in bbox)
                x_min, y_min, x_max, y_max = x1, y1, x1 + w, y1 + h
            else:
                x_min, y_min, x_max, y_max = (int(v) for v in bbox)
            draw.line([(x_min, y_min), (x_min, y_max), (x_max, y_max),
                       (x_max, y_min), (x_min, y_min)], width=1, fill=(0, 0, 255))
            text = self.label2cat[label] + f'|{label}'
            draw.text((x_min, y_min), text, (255, 255, 0), font=font)
        # image.show()
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        cv2.imshow(f'{window}', image)
        print(f'绘制完成........')
        cv2.waitKey(waitKey)
        return