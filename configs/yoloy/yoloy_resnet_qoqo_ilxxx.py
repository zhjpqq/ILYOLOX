_base_ = [
    # '../_base_/schedules/schedule_1x.py',
    '../_base_/datasets/qoqo_detection.py',
    # '../_base_/datasets/mini_detection.py',
    '../_base_/default_runtime.py']

# task settings
task = dict(
    resume_by_task=[False, 1, 2, 3][0],
    resume_by_epoch='',
    Task1={
        'load_teacher': 0,
        'load_student': 0,
        'teacher_config': '/home/zhangjp/projects/now-projects/xmdet220/configs/yoloy/yoloy_resnet_qoqo_il.py',
        'teacher_ckpt': '/home/softlink/zhjpexp/yoloy_r50_st4gb16_qoqo_il40_x0_hard/task_1_epoch_12.pth',
        'student_ckpt': '/home/softlink/zhjpexp/yoloy_r50_st4gb16_qoqo_il40_x0_hard/task_1_epoch_12.pth',
    },
    Task2={
        'load_teacher': 0,
        'load_student': 0,
        'teacher_config': '/home/zhangjp/projects/now-projects/xmdet220/configs/yoloy/yoloy_resnet_qoqo_il.py',
        'teacher_ckpt': '/home/softlink/zhjpexp/yoloy_r18_st4gb16_qoqo_il40_x1_hard/task_1_epoch_12.pth',
        'student_ckpt': '/home/softlink/zhjpexp/yoloy_r18_st4gb16_qoqo_il40_x1_hard/task_1_epoch_12.pth',
    },
)

# model settings
model = dict(
    type='YOLOY',
    # backbone=dict(
    #     type='CSPDarknet',
    #     deepen_factor=0.33,
    #     widen_factor=0.5),           # [128, 256, 512]
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
        # in_channels=[512, 1024, 2048],
        out_channels=128,
        num_csp_blocks=1,
        use_depthwise=False,
        upsample_cfg=dict(scale_factor=2, mode='nearest'),
        conv_cfg=None,
        act_cfg=dict(type='Swish'),
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001)),
    # neck=dict(
    #     type='PAFPN_IVneck',
    #     in_channels=[128, 256, 512],
    #     out_channels=128,
    #     num_csp_blocks=1,
    #     use_depthwise=False,
    #     upsample_cfg=dict(scale_factor=2, mode='nearest'),
    #     conv_cfg=None,
    #     act_cfg=dict(type='Swish'),
    #     norm_cfg=dict(type='BN', momentum=0.03, eps=0.001)),
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
        loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, reduction='mean', loss_weight=1),
        loss_obj=dict(type='CrossEntropyLoss', use_sigmoid=True, reduction='mean', loss_weight=1),
        loss_box=dict(type='L1Loss', reduction='mean', loss_weight=0),
        loss_iou=dict(type='IoULoss', mode='square', eps=1e-16, reduction='mean', loss_weight=5),
        # loss_iou=dict(type='DIoULoss', eps=1e-16, reduction='mean', loss_weight=5),
        # loss_iou=dict(type='CIoULoss', eps=1e-16, reduction='mean', loss_weight=5),
        # loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25),
        # for increment il
        reg_val={'min': 0, 'max': 16, 'num': 17, 'use_dfl': False},

        which_index='keepindex + posandprev',  # (posindex | keepindex) + (posandprev | posnoprev)
        active_score=False,
        mixxed_score=False,
        hybrid_score=False,
        active_funct='none',
        loss_distill='v0',
        alpha_distill=1,               # 蒸馏损失的整体权重，默认为1
        cates_distill='hard',     # hard + soft + obj
        locat_distill='',       # box + iou + soft + (#decode & #encode)
        feats_distill='',                # kldv
        # loss_cd_soft=dict(type='KnowledgeDistillationKLDivLoss', T=1, reduction='mean', loss_weight=1),
        loss_cd_soft=dict(type='CrossEntropyLoss', use_sigmoid=True, reduction='mean', loss_weight=1),
        # loss_cd_soft=dict(type='MSELoss', reduction='mean', loss_weight=2),
        # loss_cd_soft=dict(type='FocalLoss', use_sigmoid=True, activated=True, target_type='soft', gamma=2.0, alpha=0.25, reduction='mean', loss_weight=1),
        loss_cd_obj=dict(type='CrossEntropyLoss', use_sigmoid=True, reduction='mean', loss_weight=1),
        # loss_cd_obj=dict(type='MSELoss', reduction='mean', loss_weight=1),
        # loss_ld_soft=dict(type='KnowledgeDistillationKLDivLoss', T=2, reduction='mean', loss_weight=1),
        # loss_ld_box=dict(type='SmoothL1Loss', reduction='mean', loss_weight=1),                     # wo #decode
        loss_ld_box=dict(type='L1Loss', reduction='mean', loss_weight=0),                             # w #decode
        # loss_ld_iou=dict(type='DIoULoss', reduction='mean', loss_weight=5),                         # w #decode
        loss_ld_iou=dict(type='IoULoss', mode='square', eps=1e-16, reduction='mean', loss_weight=5),  # w #decode
        # loss_ld_iou=dict(type='KnowledgeDistillationKLDivLoss', T=2, reduction='mean', loss_weight=1),
        loss_fd=dict(type='KnowledgeDistillationKLDivLoss', T=2, reduction='mean', loss_weight=1),
    ),
    train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
    # In order to align the source code, the threshold of the val phase is
    # 0.01, and the threshold of the test phase is 0.001.
    test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65)),
    # 配置Teacher的测试输出
    teacher_test_cfg=dict(score_thr=0.3, score_thr_small=0., size_thr_big=256,
                          size_thr_small=32, nms=dict(type='nms', iou_threshold=0.65)),
)

catsplit, catload = (80, ), (1, )
# catsplit, catload = (40, 40), (1, 0)
# catsplit, catload = (50, 30), (1, 0)
# catsplit, catload = (60, 20), (1, 0)
# catsplit, catload = (70, 10), (1, 0)
# catsplit, catload = (20, 20, 20, 20), (1, 0, 0, 0)
# catsplit, catload = (15, 15, 15, 15), (1, 0, 0, 0)
# catsplit, catload = (10, 10, 10, 10), (1, 0, 0, 0)
# catsplit, catload = (5, 5, 5, 5, 5), (1, 0, 0, 0, 0)
cat_split_load = ['auto', 'manual', 'auto:任务增量训练 & manual:任务单独训练'][0]
data = dict(
    samples_per_gpu=16, workers_per_gpu=4, cat_split_load=cat_split_load,
    train=dict(test_mode=False, catsplit=catsplit, catload=catload, catpred='prev-cur', catwise=True, imgpercent=1),
    # train=dict(dataset=dict(test_mode=False, catsplit=catsplit, catload=catload, catpred='prev-cur', catwise=True, imgpercent=1)),
    val=dict(test_mode=True, catsplit=catsplit, catload=catload, catpred='prev-cur', catwise=True, imgpercent=1),
    test=dict(test_mode=True, catsplit=catsplit, catload=catload, catpred='prev-cur', catwise=True, imgpercent=1),
)
task_nums = len(data['train']['catsplit'])

# runtime
workflow = [('train', 1)]
# workflow = [('train', 1), ('val', 2)]

################## MMDet 配置 8x4_1x #################################
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
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=0.001,
    step=[8, 11])
# # learning policy  lr2
# lr_config = dict(
#     policy='YOLOY',
#     warmup='exp',
#     by_epoch=False,
#     warmup_ratio=0.01,       # warmup_start_lr = warmup_ratio * initial_lr, warmup_end_lr=initial_lr
#     warmup_iters=800,        # 1 epoch
#     warmup_by_epoch=False,   # warmup_iters指向iter//epoch
#     num_last_epochs=1,       # 最后阶段的稳定学习率
#     min_lr_ratio=0.01)       # 最后阶段的最终最小学习率，ended_lr = min_lr_ratio * initial_lr
lr_config = [lr_config] * task_nums

runner = dict(type='TaskEpochBasedRunner', max_epochs=12, max_tasks=task_nums, save_teacher=False)
runner = [runner] * task_nums

log_config = dict(interval=100)   # org 100