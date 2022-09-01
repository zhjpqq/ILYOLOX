_base_ = [
    # '../_base_/schedules/schedule_1x.py',
    '../_base_/datasets/qoqo_detection.py',
    # '../_base_/datasets/mini_detection.py',
    '../_base_/default_runtime.py']

# task settings
task = dict(
    # teacher_config='../configs/gfl/gfl_r101_fpn_mstrain_2x_coco.py'
    # teacher_ckpt='/home/softlink/Pretrained/gfl_r101_fpn_mstrain_2x_coco_20200629_200126-dd12f847.pth'
    resume_by_task=[False, 1, 2, 3][0],
    resume_by_epoch='',
    Task1={
        'load_teacher': 0,
        'load_student': 0,
        'teacher_config': '/home/zhangjp/projects/now-projects/xmdet220/configs/yoloy/yoloy_resnet_qoqo_il.py',
        # 'teacher_ckpt': '/home/softlink/zhjpexp/yoloy-r18-stst-qoqo-il20-v0/epoch_12.pth',
        # 'teacher_ckpt': '/home/softlink/zhjpexp/yoloy-r18-stst-qoqo-il40-v1/epoch_12.pth',
        # 'student_ckpt': '/home/softlink/zhjpexp/yoloy-r18-stst-qoqo-il20-v0/epoch_12.pth',
        # 'student_ckpt': '/home/softlink/zhjpexp/yoloy-r18-stst-qoqo-il40-v1/epoch_12.pth',
        'teacher_ckpt': '/home/softlink/zhjpexp/yoloy_r18_stst_qoqo_il40_v0_b16lrcos/task_1_epoch_12.pth',
        'student_ckpt': '/home/softlink/zhjpexp/yoloy_r18_stst_qoqo_il40_v0_b16lrcos/task_1_epoch_12.pth',
    },
)

# model settings
model = dict(
    type='YOLOY',
    # backbone=dict(
    #     type='CSPDarknet',
    #     deepen_factor=0.33,
    #     widen_factor=0.5),           # [128, 256, 512]
    # backbone=dict(
    #     type='ResNet',
    #     depth=50,
    #     num_stages=4,
    #     out_indices=(1, 2, 3),      # [256, 512, 1024, 2048]
    #     frozen_stages=1,
    #     norm_cfg=dict(type='BN', requires_grad=True),
    #     norm_eval=True,
    #     style='pytorch',
    #     init_cfg=dict(type='Pretrained', checkpoint='/home/softlink/Pretrained/resnet50-19c8e357.pth')),
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(1, 2, 3),  # [64, 128, 256, 512]
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='/home/softlink/Pretrained/resnet18-5c106cde.pth')),
    # t:[96, 192, 384]->96     s: [128, 256, 512]->128,
    # m:[192, 384, 768]->192,  l:[256, 512, 1024]->256
    # neck=dict(
    #     type='YOLOYPAFPN',
    #     in_channels=[128, 256, 512],
    #     out_channels=128,
    #     num_csp_blocks=1),
    neck=dict(
        type='YOLOYPAFPN',
        in_channels=[128, 256, 512],
        out_channels=128,
        num_csp_blocks=1,
        use_depthwise=False,
        upsample_cfg=dict(scale_factor=2, mode='nearest'),
        conv_cfg=None,
        act_cfg=dict(type='Swish'),
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001), ),
    # neck=dict(
    #     type='PAFPN_IVneck',
    #     in_channels=[128, 256, 512],
    #     out_channels=128,
    #     num_csp_blocks=1,
    #     use_depthwise=False,
    #     upsample_cfg=dict(scale_factor=2, mode='nearest'),
    #     conv_cfg=None,
    #     act_cfg=dict(type='Swish'),
    #     norm_cfg=dict(type='BN', momentum=0.03, eps=0.001), ),
    # bbox_head=dict(
    #     type='YOLOYHead',
    #     num_classes=80,
    #     in_channels=128,
    #     feat_channels=128),
    bbox_head=dict(
        type='YOLOYHead',
        num_classes=80,
        in_channels=128,
        feat_channels=128,
        stacked_convs=2,
        strides=[8, 16, 32],  # org 8, 16, 32，与前面fmap的levels相等
        use_depthwise=False,
        dcn_on_last_conv=False,
        conv_bias='auto',
        conv_cfg=None,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='Swish'),

        cross_xfl=False,
        learn_fgbg='foreback',  # foreground + foreback       # 分类损失FocalLoss中是否区分前背景
        prototype='',  # none + semproto_center + semproto_sample + geoproto
        # reg_val={'min': 0, 'max': 16, 'num': 17, 'usedfl': False},
        reg_val={'min': 0, 'max': 16, 'num': 17, 'activate': 'softmax', 'method': 'v1', 'usedfl': False},
        loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, reduction='mean', loss_weight=1),
        loss_obj=dict(type='CrossEntropyLoss', use_sigmoid=True, reduction='mean', loss_weight=1),
        # loss_box=dict(type='L1Loss', reduction='mean', loss_weight=1.0),
        loss_box=dict(type='SmoothL1Loss', reduction='mean', loss_weight=1),
        # loss_iou=dict(type='IoULoss', mode='square', eps=1e-16, reduction='mean', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', eps=1e-16, reduction='mean', loss_weight=5.0),      # DIoULoss or GIoULoss
        # loss_iou=dict(type='CIoULoss', eps=1e-16, reduction='mean', loss_weight=5.0),
        # loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25),
        loss_dfl=dict(type='FocalLoss', use_sigmoid=None, activated=True, gamma=2.0, alpha=0.25, reduction='mean', loss_weight=1),

        # for increment il
        alpha_distill=1,  # 蒸馏损失的整体权重，默认为1
        hybrid_score=True,
        mixed_clsloc=False,
        normal_clsscore=True,          # 部分损失函数要求归一化Sigmoid得分
        which_index='keepindex + posonly',  # (posindex | keepindex) + (posandkeep | posnokeep | posnoprev | posonly)
        loss_distill='v0',
        cates_distill='hard',                # hard + hardsoft + normsoft + soft
        locat_distill='',                    # iou + box + soft + (#decode & #encode)
        feats_distill='',                    # kldv
        loss_cd_soft=dict(type='KnowledgeDistillationKLDivLoss', T=1, reduction='mean', loss_weight=1),
        # loss_cd_soft=dict(type='MSELoss', reduction='mean', loss_weight=2),
        # loss_cd_soft=dict(type='FocalLoss', use_sigmoid=None, activated=True, gamma=2.0, alpha=0.25, reduction='mean', loss_weight=1),
        loss_ld_soft=dict(type='KnowledgeDistillationKLDivLoss', T=2, reduction='mean', loss_weight=1),
        loss_ld_bbox=dict(type='SmoothL1Loss', reduction='mean', loss_weight=2),            # wo #decode
        # loss_ld_bbox=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),   # w #decode
        # loss_ld_iou=dict(type='DIoULoss', loss_weight=1, reduction='mean'),              # w #decode
        loss_ld_iou=dict(type='GIoULoss', reduction='mean', loss_weight=2),
        # loss_fd=dict(type='KnowledgeDistillationKLDivLoss', T=2, reduction='mean', loss_weight=1),
        loss_fd=dict(type='MSELoss', reduction='mean', loss_weight=1),
    ),
    train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
    # In order to align the source code, the threshold of the val phase is
    # 0.01, and the threshold of the test phase is 0.001.
    test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65)),
    # 配置Teacher的测试输出
    teacher_test_cfg=dict(score_thr=0.3, nms=dict(type='nms', iou_threshold=0.65)),
)

# catsplit, catload = (80, ), (1, )
# catsplit, catload = (40, 40), (1, 0)
catsplit, catload = (20, 20, 20, 20), (1, 0, 0, 0)
# catsplit, catload = (15, 15, 15, 15), (1, 0, 0, 0)
# catsplit, catload = (10, 10, 10, 10), (1, 0, 0, 0)
# catsplit, catload = (5, 5, 5, 5, 5), (1, 0, 0, 0, 0)
cat_split_load = ['auto', 'manual', 'auto:任务增量训练 & manual:任务单独训练'][0]
data = dict(
    samples_per_gpu=16, workers_per_gpu=4, cat_split_load=cat_split_load,
    train=dict(test_mode=False, catsplit=catsplit, catload=catload, catpred='prev-cur', catwise=True, imgpercent=1),
    val=dict(test_mode=True, catsplit=catsplit, catload=catload, catpred='prev-cur', catwise=True, imgpercent=1),
    test=dict(test_mode=True, catsplit=catsplit, catload=catload, catpred='prev-cur', catwise=True, imgpercent=1),
)
task_nums = len(data['train']['catsplit'])

# runtime
workflow = [('train', 1)]
# workflow = [('train', 1), ('val', 1)]

################## MMDet 配置 8x4_1x ############################
# # optimizer op1 ==> schedule_1x.py
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# optimizer op2
# optimizer = dict(
#     type='SGD', lr=0.02,
#     momentum=0.9, weight_decay=0.0001, nesterov=True,
#     paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.))
# optimizer_config = dict(grad_clip=None)
# # optimizer op3
# optimizer = dict(type='AdamW', lr=0.000025, weight_decay=0.0001)
# optimizer_config = dict(grad_clip=dict(max_norm=1, norm_type=2))
optimizer = [optimizer] * task_nums

# # learning policy  lr1 ==> schedule_1x.py
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=200,
#     warmup_ratio=0.001,
#     step=[8, 11])
# # learning policy  lr2
lr_config = dict(
    policy='YOLOX',
    warmup='exp',
    by_epoch=False,
    warmup_ratio=0.01,       # warmup_start_lr = warmup_ratio * initial_lr, warmup_end_lr=initial_lr
    warmup_iters=800,        # 1 epoch
    warmup_by_epoch=False,   # warmup_iters指向iter//epoch
    num_last_epochs=1,       # 最后阶段的稳定学习率
    min_lr_ratio=0.01)       # 最后阶段的最终最小学习率，ended_lr = min_lr_ratio * initial_lr
lr_config = [lr_config] * task_nums

runner = dict(type='TaskEpochBasedRunner', max_epochs=12, max_tasks=task_nums, save_teacher=False)
runner = [runner] * task_nums


log_config = dict(
    interval=100,   # org 100
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ]
)