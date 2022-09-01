from PIL import Image
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import sys
import os.path

# config_file = './configs/fass/fass_r50_fpn_gn_1x.py'
# checkpoint_file = '../work_dirs/fass4/fass_r50_fpn_gn_1x-0125_1914-/latest.pth'

# config_file = '../configs/aaamixer/aaamixer_resnet_1x_qoqo.py'
# checkpoint_file = '/home/softlink/zhjpexp/aaamixer_r18_qoqo_stst/epoch_12.pth'

config_file = '../configs/aaamixer/aaamixer_resnet_qoqo_il.py'
checkpoint_file = '/home/softlink/zhjpexp/common_exp_il/task_1_epoch_3.pth'

# build the model from a config file and a checkpoint file
# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=(1333, 800),
#         flip=False,
#         transforms=[
#             dict(type='Resize', keep_ratio=True),
#             dict(type='RandomFlip'),
#             dict(type='Normalize', **img_norm_cfg),
#             dict(type='Pad', size_divisor=32),
#             dict(type='ImageToTensor', keys=['img']),
#             dict(type='Collect', keys=['img']),
#         ])
# ]
# data = dict(test=dict(pipeline=test_pipeline))
# cfg_options = dict(data=data)
model = init_detector(config_file, checkpoint_file, device='cuda:0')
print(f'==========\n{model}\n==========')

# test a single image and show the results
img1 = '/home/softlink/dataset/COCO2017/val2017/000000210299.jpg'   # person on bike
img2 = '/home/softlink/dataset/COCO2017/val2017/000000057597.jpg'
# img3 = '/home/softlink/dataset/COCO2017/val2017/000000577959.jpg'
# img4 = 'resources/corruptions_sev_3.png'
# img5 = 'demo/traffic.png'

# Image.open(img).save('demo_testin.jpg')
# assert False
imgs = [img1, img2][0:2]
print(imgs)
results = inference_detector(model, imgs)
# print(result)
for i, (img, res) in enumerate(zip(imgs, results)):
    file_name = img.split('/')[-1]
    file_path = f'demo/{file_name}'
    show_result_pyplot(model, img, res, score_thr=float(0.3))
