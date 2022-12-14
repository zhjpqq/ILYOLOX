# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
import torch.distributed as dist
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import get_git_hash

from mmdet import __version__
from mmdet.apis import init_random_seed, set_random_seed, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import (collect_env, get_root_logger, setup_multi_processes, update_data_root)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config',
                        # default='../configs/yolox/yolox_resnet_qoqo.py',
                        # default='../configs/ld/ld_r18_gflv1_r101_fpn_coco_1x.py',
                        # default='../configs/deformable_detr/deformdetr_resnet_qoqo.py',
                        # default='../configs/yolof/yolof_resnet_qoqo.py',
                        # default='../configs/yoloy/yoloy_resnet_qoqo.py',
                        default='../configs/aaamixer/aaamixer_resnet_qoqo.py',
                        help='train config file path')
    parser.add_argument('--work-dir',
                        # default='/home/softlink/experiments/wrxt-yoloy-r18-stst-k6u3',
                        # default='/home/softlink/zhjpexp/yolof-r18-stst-qoqo-t40',
                        # default='/home/softlink/zhjpexp/yoloy-r18-stst-qoqo',
                        default='/home/softlink/zhjpexp/common_exp',
                        # default='/home/softlink/experiments/sparse_resnet_1level_hlkt',
                        help='the dir to save logs and models')
    parser.add_argument('--resume-from',
                        # default='/home/softlink/zhjpexp/common_exp/epoch_11_ok.pth',
                        help='the checkpoint file to resume from')
    parser.add_argument('--print-model', default=False, action='store_true', help='是否打印模型结构')
    parser.add_argument('--auto-resume', action='store_true', help='resume from the latest checkpoint automatically')
    parser.add_argument('--no-validate', default=False, action='store_true',
                        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpu-id', type=int, default=[0, 1, 2, 3],
        help='id of gpu to use (only applicable to non-distributed training)')
    parser.add_argument('--seed', default=111, type=int, help='random seed')
    parser.add_argument(
        '--diff-seed', action='store_true',
        help='Whether or not set different seeds for different ranks')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both '
            'specified, --options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    # update data root according to MMDET_DATASETS
    update_data_root(cfg)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
        # 自动创建实验文件夹及日志文件
        if not osp.exists(cfg.work_dir):
            os.makedirs(cfg.work_dir)
            print(f'\n创建工作文件夹成功 => work_dir: {cfg.work_dir}')
        nohup_file = f'{cfg.work_dir}/nohup'.replace('//', '/')
        if not osp.exists(nohup_file):
            file = open(nohup_file, 'w')
            file.close()
            print(f'\n创建日志文件成功 => nohup: {nohup_file}\n')
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        # cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])
        cfg.work_dir = '/home/softlink/zhjpexp/experiments/common_exp'
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    cfg.auto_resume = args.auto_resume
    cfg.gpu_ids = [args.gpu_id]

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text
    # log some basic info
    logger.info(f'Distributed Training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    seed = init_random_seed(args.seed)
    seed = seed + dist.get_rank() if args.diff_seed else seed
    logger.info(f'Set random seed to {seed}, deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    meta['seed'] = seed
    meta['exp_name'] = osp.basename(args.config)

    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    if args.print_model:
        print('\n============== model arch start ==============')
        print(model)
        print('============== model arch end ==============\n')

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__ + get_git_hash()[:7],
            CLASSES=datasets[0].CLASSES)
    # add an attribute for visualization convenience
    model.ALL_CLASSES_IDS = datasets[0].ALL_CLASSES_IDS
    model.ALL_CLASSES = datasets[0].ALL_CLASSES
    model.PRED_CLASSES = datasets[0].PRED_CLASSES
    model.LOAD_CLASSES = datasets[0].LOAD_CLASSES
    model.START_LABEL = datasets[0].START_LABEL
    model.cat2label = datasets[0].cat2label
    model.label2cat = datasets[0].label2cat
    # print(f'\n当前CAT-LABEL：{model.cat2label}')
    train_detector(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=not args.no_validate,
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    tic = time.time()
    main()
    toc = (time.time() - tic)/60
    print(f'训练总计耗时：{toc} 分钟!')
