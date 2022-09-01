# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import datetime
import os
import os.path as osp
import time
import warnings

import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import get_git_hash

from mmdet import __version__
from mmdet.apis import init_random_seed, set_random_seed, train_detector
from mmdet.datasets import build_dataset, build_dataloader, replace_ImageToTensor
from mmdet.models import build_detector
from mmdet.utils import (collect_env, get_root_logger, setup_multi_processes,
                         update_data_root, find_latest_checkpoint)

from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (DistSamplerSeedHook, EpochBasedRunner,
                         Fp16OptimizerHook, OptimizerHook, build_optimizer,
                         build_runner, get_dist_info)
from mmcvil import TaskEpochBasedRunner

from mmdet.core import DistEvalHook, EvalHook


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config',
                        # default='../configs/yolof/yolof_resnet_qoqo_il.py',
                        # default='../configs/yoloy/yoloy_resnet_qoqo_il.py',
                        default='../configs/yoloy/yoloy_resnet_wrxt_il.py',
                        # default='../configs/aaamixer/aaamixer_resnet_qoqo_ilx.py',
                        # default='../configs/deformable_detr/deformdetr_resnet_qoqo.py',
                        help='train config file path')
    parser.add_argument('--work-dir',
                        # default='/home/softlink/zhjpexp/yoloy-r18-stst-wrxt-il54-v0',
                        # default='/home/softlink/zhjpexp/yolof-r18-stst-qoqo-il80-v1',
                        # default='/home/softlink/zhjpexp/yoloy-r18-stst-qoqo-il',
                        # default='/home/softlink/zhjpexp/aaamixer-r18-stst-qoqo-il10-v1-wodfl',
                        default='/home/softlink/zhjpexp/common_exp_il',
                        help='the dir to save logs and models')
    parser.add_argument('--resume-from', default='', help='the checkpoint file to resume from')
    parser.add_argument('--print-model', default=False, action='store_true', help='是否打印模型结构')
    parser.add_argument('--auto-resume', action='store_true',
                        help='resume from the latest checkpoint automatically')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpu-id', type=int, default=[0, 1, 2, 3],
        help='id of gpu to use (only applicable to non-distributed training)')
    parser.add_argument('--seed', default=111, type=int, help='random seed')
    parser.add_argument('--find_unused_param', default=False, action='store_true', help='参数检索')
    parser.add_argument(
        '--diff-seed', action='store_true',
        help='Whether or not set different seeds for different ranks')
    parser.add_argument(
        '--deterministic',
        default=False,
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
        default='none', help='job launcher')
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
    # 配置设定
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

    # 路径配置
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
    # log_file = osp.join(cfg.work_dir, f'nohup')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # 日志配置
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
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    seed = init_random_seed(args.seed)
    seed = seed + dist.get_rank() if args.diff_seed else seed
    logger.info(f'Set random seed to {seed}, deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    meta['seed'] = seed
    meta['exp_name'] = osp.basename(args.config)

    # 任务设定
    task_nums = len(cfg.data.train.catsplit)
    task_list = [f'Task{i + 1}' for i in range(task_nums)]
    task_index = list(range(1, task_nums + 1))
    if cfg.data.cat_split_load == 'auto':     # 各任务连续自动训练
        task_train_catload = np.identity(task_nums).astype(np.int8).tolist()
        task_val_catload = np.tril(np.ones(task_nums), k=0).astype(np.int8).tolist()
    elif cfg.data.cat_split_load == 'manual':    # 指定某个任务单独训练
        task_train_catload = [cfg.data.train.catload for _ in range(task_nums)]
        task_val_catload = [cfg.data.val.catload for _ in range(task_nums)]
    else:
        raise NotImplementedError(f'{cfg.data.cat_split_load}')

    # 任务循环
    model = None
    for tid, task in zip(task_index, task_list):
        print(f'\n\n======== Task-{tid}({task}) 开始, At {datetime.datetime.now()} ==========')
        if cfg.data.cat_split_load == 'manual' and task_train_catload[tid-1][tid-1] == 0:
            # 单独训练catload中指定的第几个任务
            print(f'\n\n======== Task-{tid}({task}) 跳过, 指定任务单独训练, At {datetime.datetime.now()} ==========')
            model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
            model.init_weights()
            continue
        if tid < cfg.task.resume_by_task:
            # 跳转到指定任务恢复训练
            print(f'\n\n======== Task-{tid}({task}) 跳过, 跳转到指定任务恢复训练, At {datetime.datetime.now()} ==========')
            continue
        tic_run = time.time()

        # 模型加载
        if tid == 1:
            teacher_config, teacher_ckpt, student_ckpt = None, None, None
            model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
            model.init_weights()
            if cfg.task[task]['load_student']:
                student_ckpt = cfg.task[task]['student_ckpt']
                model.set_student(ckptfile=student_ckpt)
            if cfg.task[task]['load_teacher']:
                teacher_config = cfg.task[task]['teacher_config']
                teacher_ckpt = cfg.task[task]['teacher_ckpt']
            # 任务1一般无需教师模型，若需要则必须从外部配置，自己无法生成教师模型！
            model.set_teacher(config=teacher_config, ckptfile=teacher_ckpt, model=None, trainval='val', taskid=tid)
            if tid == cfg.task.resume_by_task:
                print(f'\n====== Task-{tid}({task}) 跳过，Resume From Prev-Student-Model CheckPoint: {student_ckpt}')
                continue
        else:
            if tid == cfg.task.resume_by_task:
                model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
                model.init_weights()
                assert cfg.task[task]['load_student'] and cfg.task[task]['student_ckpt'], print('从上次任务恢复启动，必须载入上次任务的模型权值！')
                model.set_student(ckptfile=cfg.task[task]['student_ckpt'])
                print(f'\n====== Task-{tid}({task}) 跳过，Resume From CheckPoint: {cfg.task[task]["student_ckpt"]}')
                continue
            else:
                if isinstance(model, (MMDataParallel, MMDistributedDataParallel)):
                    model = model.module
                # 通过teacher_config和teacher_ckpt参数可手动为某次任务配置教师模型，否则自动使用上次任务模型作为下次任务的教师；
                # 一般无需手动配置，直接使用上一任务的model作为下一任务的教师模型。
                teacher_config, teacher_ckpt = None, None
                if getattr(cfg.task, task, None) and getattr(cfg.task[task], 'load_teacher', False):
                    teacher_config = cfg.task[task]['teacher_config']
                    teacher_ckpt = cfg.task[task]['teacher_ckpt']
                model.set_teacher(config=teacher_config, ckptfile=teacher_ckpt, trainval='val',
                                  model=copy.deepcopy(model) if teacher_ckpt is None else None, taskid=tid)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('Param Nums of Model:', n_parameters)

        # 数据加载
        cfg.data.train.update({'catsplit': cfg.data.train.catsplit, 'catload': task_train_catload[tid - 1]})
        cfg.data.val.update({'catsplit': cfg.data.val.catsplit, 'catload': task_val_catload[tid - 1]})
        train_dataset = build_dataset(cfg.data.train, dict(test_mode=False))
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        train_dataloader = build_dataloader(
            train_dataset, cfg.data.samples_per_gpu, cfg.data.workers_per_gpu,
            num_gpus=len(cfg.gpu_ids), dist=distributed, seed=cfg.seed,
            runner_type=cfg.runner[tid - 1]['type'],
            persistent_workers=cfg.data.get('persistent_workers', False))
        val_dataloader = build_dataloader(
            val_dataset, cfg.data.val.pop('samples_per_gpu', 1), cfg.data.workers_per_gpu,
            dist=distributed, shuffle=False)
        model.set_datainfo(cat2id=train_dataset.ALL_CLASSES_IDS,
                           cat2label=train_dataset.cat2label,
                           pred_cat=train_dataset.PRED_CLASSES,    # PREV + CUR
                           load_cat=train_dataset.LOAD_CLASSES,    # CUR
                           task_cat=train_dataset.TASK_CLASSES,)   # PREV + CUR + NEXT
        print(f'\nTask-{tid} 训练图像数量: {len(train_dataset)} & 验证图像数量: {len(val_dataset)}\n')
        # print(f'训练集加载类别==> {len(train_dataset.LOAD_CLASSES)}', train_dataset.LOAD_CLASSES)
        # print(f'验证集加载类别==> {len(val_dataset.LOAD_CLASSES)}', val_dataset.LOAD_CLASSES)
        # # # 数据检查
        # check_dataloader, check_dataset = train_dataloader, train_dataset
        # # check_dataloader, check_dataset = val_dataloader, val_dataset     # TODO val-wo-gtbbox&gtlabel ??
        # for batch_idx, batch_data in enumerate(check_dataloader):
        #     # 检查加载数据的(类别Name、类别ID、类别Label)是否一一匹配和正确！
        #     print(f"Task({tid}) -- Batch({batch_idx})")
        #     gt_imgs_batch = batch_data['img_metas'].data[0]
        #     gt_labels_batch = [l.numpy().tolist() for l in batch_data['gt_labels'].data[0]]
        #     gt_bboxes_batch = [l.numpy().tolist() for l in batch_data['gt_bboxes'].data[0]]
        #     for gt_labels in gt_labels_batch:
        #         gt_cat_ids = [check_dataset.label2cat[l] for l in gt_labels]
        #         filter_catid = [catid in check_dataset.cat_ids_load for catid in gt_cat_ids]
        #         assert all(filter_catid), print(f'Task({tid}) filter_catid=> ', filter_catid)
        #         gt_cat_names = [check_dataset.ALL_IDS_CLASSES[catid] for catid in gt_cat_ids]
        #         filter_catname = [catname in check_dataset.LOAD_CLASSES for catname in gt_cat_names]
        #         assert all(filter_catname), print(f'Task({tid}) filter_catname=> ', filter_catname)
        #     for idx, (imgmeta, gtlabels, gtboxes) in enumerate(zip(gt_imgs_batch, gt_labels_batch, gt_bboxes_batch)):
        #         check_dataset.draw_boxes_on_img_v1(img_info=imgmeta, labels=gtlabels, boxes=gtboxes,
        #                                            coord='x1y1x2y2', isnorm=False, waitKey=-200,
        #                                            window='imgshow', realtodo=1)
        # continue

        # 模型训练  # put model on gpus
        if distributed:
            # Sets the `find_unused_parameters` parameter in torch.nn.parallel.DistributedDataParallel
            model = MMDistributedDataParallel(
                model.cuda(), device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False, find_unused_parameters=args.find_unused_param)
        else:
            model = MMDataParallel(model, device_ids=cfg.gpu_ids)

        if isinstance(model, (MMDataParallel, MMDistributedDataParallel)):
            if model.module.has_teacher:
                model.module.teacher_model.cuda()

        # build optimizer
        # continue
        optimizer = build_optimizer(model, cfg.optimizer[tid - 1])

        # build lr_schedule
        lr_config = cfg.lr_config[tid - 1]

        # build runner
        runner = build_runner(
            cfg.runner[tid - 1],
            default_args=dict(
                model=model,
                optimizer=optimizer,
                work_dir=cfg.work_dir,
                logger=logger,
                meta=meta))
        runner.timestamp = timestamp

        # fp16 setting
        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            optimizer_config = Fp16OptimizerHook(**cfg.optimizer_config, **fp16_cfg, distributed=distributed)
        elif distributed and 'type' not in cfg.optimizer_config:
            optimizer_config = OptimizerHook(**cfg.optimizer_config)
        else:
            optimizer_config = cfg.optimizer_config

        # register hooks
        runner.register_training_hooks(
            lr_config,
            optimizer_config,
            cfg.checkpoint_config,
            cfg.log_config,
            cfg.get('momentum_config', None),
            custom_hooks_config=cfg.get('custom_hooks', None))
        if distributed:
            if isinstance(runner, TaskEpochBasedRunner):
                runner.register_hook(DistSamplerSeedHook())
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner[tid - 1]['type'] != 'IterBasedRunner'
        eval_hook = DistEvalHook if distributed else EvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg), priority='LOW')

        # cfg.resume_from = cfg.task.resume_by_epoch or cfg.resume_from
        if cfg.task.resume_by_epoch:
            print(f'\n在任务 {task} 的 XX回合上加载学生模型！'
                  f'cfg.task.resume_by_epoch={cfg.task.resume_by_epoch}')
            cfg.resume_from = cfg.task.resume_by_epoch
        # TODO 下方传入参数控制Teacher模型的加载
        if cfg.resume_from:
            runner.resume(cfg.resume_from)
        elif cfg.load_from:
            runner.load_checkpoint(cfg.load_from)
        runner.run([train_dataloader], cfg.workflow, cur_task=tid)
        toc_run = (time.time() - tic_run) / 60
        print(f'{task} 训练总计耗时：{toc_run} 分钟!')


if __name__ == '__main__':
    tic = time.time()
    main()
    toc = (time.time() - tic) / 60
    print(f'训练总计耗时：{toc} 分钟!')
