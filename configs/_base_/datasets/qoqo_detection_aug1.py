# dataset settings
dataset_type = 'CocoDataset'
data_root = '/home/softlink/dataset/COCO2017/'

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# TODO 按比例缩减图像尺寸为原来的一半 666/400 = 666.5/400 = 1.66625，均值方差重新计算？？
# TODO 缺400~666的正方形，比例覆盖不完全，是否影响检测？？
# 训练设置
img_scale_size0 = [416, 448, 480, 512, 544, 576, 608, 640]
img_scale_size1 = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640]  # average_area_ratio=0.75  # 用于多尺度训练，value模式
img_scale_size2 = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]  # average_area_ratio=1.0
img_scale_size3 = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]  # average_area_ratio=1.33 640±160=[480, 800]
# img_scale_size = (640, 640)  # img_scale_size1        # TODO TODO TODO TODO 多尺度接触封印
# 使用方法1：img_scale_size=[(640, value) for value in img_scale_size],                # 尺寸大部分集中于min/max中.
# 使用方法2：img_scale_size=[(w, h) for w in img_scale_size for h in img_scale_size],  # 尺寸分散于所有尺寸组合中.
# 测试设置
max_min_size = (640, 640)  # 用于测试，==min(大/长，小/短)*(长，短)，小值不可太小，否则小/短始终小于大/长，应该≥512！
# ① keep_ratio=False,则转换后尺寸为: 固定的 (W, H) = (max_min_size[0], max_min_size[1])
# ② keep_ratio=True, 则转换后尺寸为: 当(大/长，小/短)缩放比接近时 max(W, H)<=max_size, min(W,H)可＜min_size!
#                                  当(大/长，小/短)缩放比过大时 min(W, H)>=min_size, 但max(W, H)不可＞max_size!
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize',
         img_scale=[(w, h) for w in img_scale_size0 for h in img_scale_size0],
         multiscale_mode='value',
         keep_ratio=True),
    dict(type='PhotoMetricDistortion',
         brightness_delta=32,
         contrast_range=(0.5, 1.5),
         saturation_range=(0.5, 1.5),
         hue_delta=18),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='MultiScaleFlipAug',
         img_scale=max_min_size,
         flip=False,
         transforms=[
             dict(type='Resize', multiscale_mode='range', keep_ratio=True),
             dict(type='RandomFlip'),
             dict(type='Normalize', **img_norm_cfg),
             dict(type='Pad', size_divisor=32),
             dict(type='ImageToTensor', keys=['img']),
             dict(type='Collect', keys=['img']),
         ])
]

which = ['', '_mini1k', '_mini2k', '_mini5k', '_mini2w', '_mini3w', '_mini5w']
trainwhich = ''
valwhich = ''

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + f'annotations/instances_train2017{trainwhich}.json',
        img_prefix=data_root + 'train2017/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + f'annotations/instances_val2017{valwhich}.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + f'annotations/instances_val2017{valwhich}.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
