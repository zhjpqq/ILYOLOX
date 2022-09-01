def __get_debug():
    import os
    return 'C_DEBUG' in os.environ

debug = __get_debug()

log_interval = 100

_base_ = [
    # '../_base_/datasets/coco_detection.py',
    '../_base_/datasets/qoqo_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

# model settings
num_stages = 6      # org 6
num_query = 100
QUERY_DIM = 256
FEAT_DIM = 256
FF_DIM = 2048       # org 2048

# P_in for spatial mixing in the paper.
in_points_list = [32, ] * num_stages

# P_out for spatial mixing in the paper. Also named as `out_points` in this codebase.
out_patterns_list = [128, ] * num_stages

# G for the mixer grouping in the paper. Please distinguishe it from num_heads in MHSA in this codebase.
n_group_list = [4, ] * num_stages

model = dict(
    type='QueryBased',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),  # [256, 512, 1024, 2048]
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='/home/softlink/Pretrained/resnet50-19c8e357.pth')),
    # backbone=dict(
    #     type='ResNet',
    #     depth=18,
    #     num_stages=4,
    #     out_indices=(0, 1, 2, 3),  # [64, 128, 256, 512]
    #     frozen_stages=1,
    #     norm_cfg=dict(type='BN', requires_grad=True),
    #     norm_eval=True,
    #     style='pytorch',
    #     init_cfg=dict(type='Pretrained', checkpoint='/home/softlink/Pretrained/resnet18-5c106cde.pth')),
    neck=dict(
        type='ChannelMapping',
        in_channels=[256, 512, 1024, 2048],
        out_channels=FEAT_DIM,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=4),
    rpn_head=dict(
        type='InitialQueryGenerator',
        method='xyzr',
        num_query=num_query,
        content_dim=QUERY_DIM),
    roi_head=dict(
        type='AdaMixerDecoder',
        featmap_strides=[4, 8, 16, 32],
        num_stages=num_stages,
        stage_loss_weights=[1] * num_stages,
        content_dim=QUERY_DIM,
        bbox_head=[
            dict(
                type='AdaMixerDecoderStage',
                num_classes=80,
                num_ffn_fcs=2,
                num_heads=8,
                num_cls_fcs=1,
                num_reg_fcs=1,
                feedforward_channels=FF_DIM,
                content_dim=QUERY_DIM,
                feat_channels=FEAT_DIM,
                dropout=0.0,
                in_points=in_points_list[stage_idx],
                out_points=out_patterns_list[stage_idx],
                n_groups=n_group_list[stage_idx],
                ffn_act_cfg=dict(type='ReLU', inplace=True),
                loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=2.0),
                # NOTE: The following argument is a placeholder to hack the code. No real effects for decoding or updating bounding boxes.
                # bbox_coder=dict(
                #     type='DeltaXYWHBBoxCoder',
                #     clip_border=False,
                #     target_means=[0., 0., 0., 0.],
                #     target_stds=[0.5, 0.5, 1., 1.]),
            ) for stage_idx in range(num_stages)
        ]),
    # training and testing settings
    train_cfg=dict(
        rpn=None,
        rcnn=[
            dict(
                assigner=dict(
                    type='HungarianAssigner',
                    cls_cost=dict(type='FocalLossCost', weight=2.0),
                    reg_cost=dict(type='BBoxL1Cost', weight=5.0),
                    iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0)),
                sampler=dict(type='PseudoSampler'),
                pos_weight=1) for _ in range(num_stages)
        ]),
    test_cfg=dict(rpn=None, rcnn=dict(max_per_img=num_query)))

# dataset settings
data = dict(
    samples_per_gpu=16, workers_per_gpu=4,
    train=dict(catwise=True, imgpercent=1),
)

# runtime
workflow = [('train', 1)]
# workflow = [('train', 1), ('val', 1)]

# optimizer
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.000025,
    weight_decay=0.0001,
)
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=1.0, norm_type=2))

# learning policy
lr_config = dict(
    policy='step',
    step=[8, 11],
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001
)
runner = dict(type='EpochBasedRunner', max_epochs=12)


def __date():
    import datetime
    return datetime.datetime.now().strftime('%m%d_%H%M')


log_config = dict(
    interval=log_interval,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ]
)

postfix = '_' + __date()

find_unused_parameters = True

resume_from = None
