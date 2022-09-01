/home/zhangjp/miniconda3/envs/pth18/bin/python /home/zhangjp/project/xmdet216/tools/misc/check_model_arch.py

输入参数: Namespace(checkpoint=None, config='../../configs/deformable_detr/deformable_detr_r50_1x_hlkt.py', device='cuda:1', score_thr=0.3)

########### Model Configs #################


########### Model Architecture ############

/home/zhangjp/miniconda3/envs/pth18/lib/python3.7/site-packages/mmcv/cnn/bricks/transformer.py:349: UserWarning: The arguments `feedforward_channels` in BaseTransformerLayer has been deprecated, now you should set `feedforward_channels` and other FFN related arguments to a dict named `ffn_cfgs`. 
  f'The arguments `{ori_name}` in BaseTransformerLayer '
/home/zhangjp/miniconda3/envs/pth18/lib/python3.7/site-packages/mmcv/cnn/bricks/transformer.py:349: UserWarning: The arguments `ffn_dropout` in BaseTransformerLayer has been deprecated, now you should set `ffn_drop` and other FFN related arguments to a dict named `ffn_cfgs`. 
  f'The arguments `{ori_name}` in BaseTransformerLayer '
/home/zhangjp/miniconda3/envs/pth18/lib/python3.7/site-packages/mmcv/cnn/bricks/transformer.py:349: UserWarning: The arguments `ffn_num_fcs` in BaseTransformerLayer has been deprecated, now you should set `num_fcs` and other FFN related arguments to a dict named `ffn_cfgs`. 
  f'The arguments `{ori_name}` in BaseTransformerLayer '
/home/zhangjp/miniconda3/envs/pth18/lib/python3.7/site-packages/mmcv/cnn/bricks/transformer.py:92: UserWarning: The arguments `dropout` in MultiheadAttention has been deprecated, now you can separately set `attn_drop`(float), proj_drop(float), and `dropout_layer`(dict) 
  warnings.warn('The arguments `dropout` in MultiheadAttention '
DeformableDETR(
  (backbone): ResNet(
    (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu): ReLU(inplace=True)
    (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    (layer1): ResLayer(
      init_cfg={'type': 'Constant', 'val': 0, 'override': {'name': 'norm3'}}
      (2): Bottleneck(
        (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
      init_cfg={'type': 'Constant', 'val': 0, 'override': {'name': 'norm3'}}
    )
    (layer4): ResLayer(
      init_cfg={'type': 'Constant', 'val': 0, 'override': {'name': 'norm3'}}
      (2): Bottleneck(
        (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
      init_cfg={'type': 'Constant', 'val': 0, 'override': {'name': 'norm3'}}
    )
  )
  init_cfg=[{'type': 'Kaiming', 'layer': 'Conv2d'}, {'type': 'Constant', 'val': 1, 'layer': ['_BatchNorm', 'GroupNorm']}]
  (neck): ChannelMapper(
    (convs): ModuleList(
      (0): ConvModule(
        (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (gn): GroupNorm(32, 256, eps=1e-05, affine=True)
      )
      (1): ConvModule(
        (conv): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (gn): GroupNorm(32, 256, eps=1e-05, affine=True)
      )
      (2): ConvModule(
        (conv): Conv2d(2048, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (gn): GroupNorm(32, 256, eps=1e-05, affine=True)
      )
    )
    (extra_convs): ModuleList(
      (0): ConvModule(
        (conv): Conv2d(2048, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (gn): GroupNorm(32, 256, eps=1e-05, affine=True)
      )
    )
  )
  init_cfg={'type': 'Xavier', 'layer': 'Conv2d', 'distribution': 'uniform'}
  (bbox_head): DeformableDETRHead(
    (loss_cls): FocalLoss()
    (loss_bbox): L1Loss()
    (loss_iou): GIoULoss()
    (activate): ReLU(inplace=True)
    (positional_encoding): SinePositionalEncoding(num_feats=128, temperature=10000, normalize=True, scale=6.283185307179586, eps=1e-06)
    (transformer): DeformableDetrTransformer(
      (encoder): DetrTransformerEncoder(
        (layers): ModuleList(
          (0): BaseTransformerLayer(
            (attentions): ModuleList(
              (0): MultiScaleDeformableAttention(
                (dropout): Dropout(p=0.1, inplace=False)
                (sampling_offsets): Linear(in_features=256, out_features=256, bias=True)
                (attention_weights): Linear(in_features=256, out_features=128, bias=True)
                (value_proj): Linear(in_features=256, out_features=256, bias=True)
                (output_proj): Linear(in_features=256, out_features=256, bias=True)
              )
            )
            (ffns): ModuleList(
              (0): FFN(
                (activate): ReLU(inplace=True)
                (layers): Sequential(
                  (0): Sequential(
                    (0): Linear(in_features=256, out_features=1024, bias=True)
                    (1): ReLU(inplace=True)
                    (2): Dropout(p=0.1, inplace=False)
                  )
                  (1): Linear(in_features=1024, out_features=256, bias=True)
                  (2): Dropout(p=0.1, inplace=False)
                )
                (dropout_layer): Identity()
              )
            )
            (norms): ModuleList(
              (0): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
              (1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            )
          )
          (1): BaseTransformerLayer(
            (attentions): ModuleList(
              (0): MultiScaleDeformableAttention(
                (dropout): Dropout(p=0.1, inplace=False)
                (sampling_offsets): Linear(in_features=256, out_features=256, bias=True)
                (attention_weights): Linear(in_features=256, out_features=128, bias=True)
                (value_proj): Linear(in_features=256, out_features=256, bias=True)
                (output_proj): Linear(in_features=256, out_features=256, bias=True)
              )
            )
            (ffns): ModuleList(
              (0): FFN(
                (activate): ReLU(inplace=True)
                (layers): Sequential(
                  (0): Sequential(
                    (0): Linear(in_features=256, out_features=1024, bias=True)
                    (1): ReLU(inplace=True)
                    (2): Dropout(p=0.1, inplace=False)
                  )
                  (1): Linear(in_features=1024, out_features=256, bias=True)
                  (2): Dropout(p=0.1, inplace=False)
                )
                (dropout_layer): Identity()
              )
            )
            (norms): ModuleList(
              (0): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
              (1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            )
          )
          (2): BaseTransformerLayer(
            (attentions): ModuleList(
              (0): MultiScaleDeformableAttention(
                (dropout): Dropout(p=0.1, inplace=False)
                (sampling_offsets): Linear(in_features=256, out_features=256, bias=True)
                (attention_weights): Linear(in_features=256, out_features=128, bias=True)
                (value_proj): Linear(in_features=256, out_features=256, bias=True)
                (output_proj): Linear(in_features=256, out_features=256, bias=True)
              )
            )
            (ffns): ModuleList(
              (0): FFN(
                (activate): ReLU(inplace=True)
                (layers): Sequential(
                  (0): Sequential(
                    (0): Linear(in_features=256, out_features=1024, bias=True)
                    (1): ReLU(inplace=True)
                    (2): Dropout(p=0.1, inplace=False)
                  )
                  (1): Linear(in_features=1024, out_features=256, bias=True)
                  (2): Dropout(p=0.1, inplace=False)
                )
                (dropout_layer): Identity()
              )
            )
            (norms): ModuleList(
              (0): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
              (1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            )
          )
          (3): BaseTransformerLayer(
            (attentions): ModuleList(
              (0): MultiScaleDeformableAttention(
                (dropout): Dropout(p=0.1, inplace=False)
                (sampling_offsets): Linear(in_features=256, out_features=256, bias=True)
                (attention_weights): Linear(in_features=256, out_features=128, bias=True)
                (value_proj): Linear(in_features=256, out_features=256, bias=True)
                (output_proj): Linear(in_features=256, out_features=256, bias=True)
              )
            )
            (ffns): ModuleList(
              (0): FFN(
                (activate): ReLU(inplace=True)
                (layers): Sequential(
                  (0): Sequential(
                    (0): Linear(in_features=256, out_features=1024, bias=True)
                    (1): ReLU(inplace=True)
                    (2): Dropout(p=0.1, inplace=False)
                  )
                  (1): Linear(in_features=1024, out_features=256, bias=True)
                  (2): Dropout(p=0.1, inplace=False)
                )
                (dropout_layer): Identity()
              )
            )
            (norms): ModuleList(
              (0): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
              (1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            )
          )
          (4): BaseTransformerLayer(
            (attentions): ModuleList(
              (0): MultiScaleDeformableAttention(
                (dropout): Dropout(p=0.1, inplace=False)
                (sampling_offsets): Linear(in_features=256, out_features=256, bias=True)
                (attention_weights): Linear(in_features=256, out_features=128, bias=True)
                (value_proj): Linear(in_features=256, out_features=256, bias=True)
                (output_proj): Linear(in_features=256, out_features=256, bias=True)
              )
            )
            (ffns): ModuleList(
              (0): FFN(
                (activate): ReLU(inplace=True)
                (layers): Sequential(
                  (0): Sequential(
                    (0): Linear(in_features=256, out_features=1024, bias=True)
                    (1): ReLU(inplace=True)
                    (2): Dropout(p=0.1, inplace=False)
                  )
                  (1): Linear(in_features=1024, out_features=256, bias=True)
                  (2): Dropout(p=0.1, inplace=False)
                )
                (dropout_layer): Identity()
              )
            )
            (norms): ModuleList(
              (0): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
              (1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            )
          )
          (5): BaseTransformerLayer(
            (attentions): ModuleList(
              (0): MultiScaleDeformableAttention(
                (dropout): Dropout(p=0.1, inplace=False)
                (sampling_offsets): Linear(in_features=256, out_features=256, bias=True)
                (attention_weights): Linear(in_features=256, out_features=128, bias=True)
                (value_proj): Linear(in_features=256, out_features=256, bias=True)
                (output_proj): Linear(in_features=256, out_features=256, bias=True)
              )
            )
            (ffns): ModuleList(
              (0): FFN(
                (activate): ReLU(inplace=True)
                (layers): Sequential(
                  (0): Sequential(
                    (0): Linear(in_features=256, out_features=1024, bias=True)
                    (1): ReLU(inplace=True)
                    (2): Dropout(p=0.1, inplace=False)
                  )
                  (1): Linear(in_features=1024, out_features=256, bias=True)
                  (2): Dropout(p=0.1, inplace=False)
                )
                (dropout_layer): Identity()
              )
            )
            (norms): ModuleList(
              (0): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
              (1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            )
          )
        )
      )
      (decoder): DeformableDetrTransformerDecoder(
        (layers): ModuleList(
          (0): DetrTransformerDecoderLayer(
            (attentions): ModuleList(
              (0): MultiheadAttention(
                (attn): MultiheadAttention(
                  (out_proj): _LinearWithBias(in_features=256, out_features=256, bias=True)
                )
                (proj_drop): Dropout(p=0.0, inplace=False)
                (dropout_layer): Dropout(p=0.1, inplace=False)
              )
              (1): MultiScaleDeformableAttention(
                (dropout): Dropout(p=0.1, inplace=False)
                (sampling_offsets): Linear(in_features=256, out_features=256, bias=True)
                (attention_weights): Linear(in_features=256, out_features=128, bias=True)
                (value_proj): Linear(in_features=256, out_features=256, bias=True)
                (output_proj): Linear(in_features=256, out_features=256, bias=True)
              )
            )
            (ffns): ModuleList(
              (0): FFN(
                (activate): ReLU(inplace=True)
                (layers): Sequential(
                  (0): Sequential(
                    (0): Linear(in_features=256, out_features=1024, bias=True)
                    (1): ReLU(inplace=True)
                    (2): Dropout(p=0.1, inplace=False)
                  )
                  (1): Linear(in_features=1024, out_features=256, bias=True)
                  (2): Dropout(p=0.1, inplace=False)
                )
                (dropout_layer): Identity()
              )
            )
            (norms): ModuleList(
              (0): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
              (1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
              (2): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            )
          )
          (1): DetrTransformerDecoderLayer(
            (attentions): ModuleList(
              (0): MultiheadAttention(
                (attn): MultiheadAttention(
                  (out_proj): _LinearWithBias(in_features=256, out_features=256, bias=True)
                )
                (proj_drop): Dropout(p=0.0, inplace=False)
                (dropout_layer): Dropout(p=0.1, inplace=False)
              )
              (1): MultiScaleDeformableAttention(
                (dropout): Dropout(p=0.1, inplace=False)
                (sampling_offsets): Linear(in_features=256, out_features=256, bias=True)
                (attention_weights): Linear(in_features=256, out_features=128, bias=True)
                (value_proj): Linear(in_features=256, out_features=256, bias=True)
                (output_proj): Linear(in_features=256, out_features=256, bias=True)
              )
            )
            (ffns): ModuleList(
              (0): FFN(
                (activate): ReLU(inplace=True)
                (layers): Sequential(
                  (0): Sequential(
                    (0): Linear(in_features=256, out_features=1024, bias=True)
                    (1): ReLU(inplace=True)
                    (2): Dropout(p=0.1, inplace=False)
                  )
                  (1): Linear(in_features=1024, out_features=256, bias=True)
                  (2): Dropout(p=0.1, inplace=False)
                )
                (dropout_layer): Identity()
              )
            )
            (norms): ModuleList(
              (0): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
              (1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
              (2): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            )
          )
          (2): DetrTransformerDecoderLayer(
            (attentions): ModuleList(
              (0): MultiheadAttention(
                (attn): MultiheadAttention(
                  (out_proj): _LinearWithBias(in_features=256, out_features=256, bias=True)
                )
                (proj_drop): Dropout(p=0.0, inplace=False)
                (dropout_layer): Dropout(p=0.1, inplace=False)
              )
              (1): MultiScaleDeformableAttention(
                (dropout): Dropout(p=0.1, inplace=False)
                (sampling_offsets): Linear(in_features=256, out_features=256, bias=True)
                (attention_weights): Linear(in_features=256, out_features=128, bias=True)
                (value_proj): Linear(in_features=256, out_features=256, bias=True)
                (output_proj): Linear(in_features=256, out_features=256, bias=True)
              )
            )
            (ffns): ModuleList(
              (0): FFN(
                (activate): ReLU(inplace=True)
                (layers): Sequential(
                  (0): Sequential(
                    (0): Linear(in_features=256, out_features=1024, bias=True)
                    (1): ReLU(inplace=True)
                    (2): Dropout(p=0.1, inplace=False)
                  )
                  (1): Linear(in_features=1024, out_features=256, bias=True)
                  (2): Dropout(p=0.1, inplace=False)
                )
                (dropout_layer): Identity()
              )
            )
            (norms): ModuleList(
              (0): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
              (1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
              (2): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            )
          )
          (3): DetrTransformerDecoderLayer(
            (attentions): ModuleList(
              (0): MultiheadAttention(
                (attn): MultiheadAttention(
                  (out_proj): _LinearWithBias(in_features=256, out_features=256, bias=True)
                )
                (proj_drop): Dropout(p=0.0, inplace=False)
                (dropout_layer): Dropout(p=0.1, inplace=False)
              )
              (1): MultiScaleDeformableAttention(
                (dropout): Dropout(p=0.1, inplace=False)
                (sampling_offsets): Linear(in_features=256, out_features=256, bias=True)
                (attention_weights): Linear(in_features=256, out_features=128, bias=True)
                (value_proj): Linear(in_features=256, out_features=256, bias=True)
                (output_proj): Linear(in_features=256, out_features=256, bias=True)
              )
            )
            (ffns): ModuleList(
              (0): FFN(
                (activate): ReLU(inplace=True)
                (layers): Sequential(
                  (0): Sequential(
                    (0): Linear(in_features=256, out_features=1024, bias=True)
                    (1): ReLU(inplace=True)
                    (2): Dropout(p=0.1, inplace=False)
                  )
                  (1): Linear(in_features=1024, out_features=256, bias=True)
                  (2): Dropout(p=0.1, inplace=False)
                )
                (dropout_layer): Identity()
              )
            )
            (norms): ModuleList(
              (0): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
              (1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
              (2): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            )
          )
          (4): DetrTransformerDecoderLayer(
            (attentions): ModuleList(
              (0): MultiheadAttention(
                (attn): MultiheadAttention(
                  (out_proj): _LinearWithBias(in_features=256, out_features=256, bias=True)
                )
                (proj_drop): Dropout(p=0.0, inplace=False)
                (dropout_layer): Dropout(p=0.1, inplace=False)
              )
              (1): MultiScaleDeformableAttention(
                (dropout): Dropout(p=0.1, inplace=False)
                (sampling_offsets): Linear(in_features=256, out_features=256, bias=True)
                (attention_weights): Linear(in_features=256, out_features=128, bias=True)
                (value_proj): Linear(in_features=256, out_features=256, bias=True)
                (output_proj): Linear(in_features=256, out_features=256, bias=True)
              )
            )
            (ffns): ModuleList(
              (0): FFN(
                (activate): ReLU(inplace=True)
                (layers): Sequential(
                  (0): Sequential(
                    (0): Linear(in_features=256, out_features=1024, bias=True)
                    (1): ReLU(inplace=True)
                    (2): Dropout(p=0.1, inplace=False)
                  )
                  (1): Linear(in_features=1024, out_features=256, bias=True)
                  (2): Dropout(p=0.1, inplace=False)
                )
                (dropout_layer): Identity()
              )
            )
            (norms): ModuleList(
              (0): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
              (1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
              (2): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            )
          )
          (5): DetrTransformerDecoderLayer(
            (attentions): ModuleList(
              (0): MultiheadAttention(
                (attn): MultiheadAttention(
                  (out_proj): _LinearWithBias(in_features=256, out_features=256, bias=True)
                )
                (proj_drop): Dropout(p=0.0, inplace=False)
                (dropout_layer): Dropout(p=0.1, inplace=False)
              )
              (1): MultiScaleDeformableAttention(
                (dropout): Dropout(p=0.1, inplace=False)
                (sampling_offsets): Linear(in_features=256, out_features=256, bias=True)
                (attention_weights): Linear(in_features=256, out_features=128, bias=True)
                (value_proj): Linear(in_features=256, out_features=256, bias=True)
                (output_proj): Linear(in_features=256, out_features=256, bias=True)
              )
            )
            (ffns): ModuleList(
              (0): FFN(
                (activate): ReLU(inplace=True)
                (layers): Sequential(
                  (0): Sequential(
                    (0): Linear(in_features=256, out_features=1024, bias=True)
                    (1): ReLU(inplace=True)
                    (2): Dropout(p=0.1, inplace=False)
                  )
                  (1): Linear(in_features=1024, out_features=256, bias=True)
                  (2): Dropout(p=0.1, inplace=False)
                )
                (dropout_layer): Identity()
              )
            )
            (norms): ModuleList(
              (0): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
              (1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
              (2): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            )
          )
        )
      )
      (reference_points): Linear(in_features=256, out_features=2, bias=True)
    )
    (cls_branches): ModuleList(
      (0): Linear(in_features=256, out_features=10, bias=True)
      (1): Linear(in_features=256, out_features=10, bias=True)
      (2): Linear(in_features=256, out_features=10, bias=True)
      (3): Linear(in_features=256, out_features=10, bias=True)
      (4): Linear(in_features=256, out_features=10, bias=True)
      (5): Linear(in_features=256, out_features=10, bias=True)
    )
    (reg_branches): ModuleList(
      (0): Sequential(
        (0): Linear(in_features=256, out_features=256, bias=True)
        (1): ReLU()
        (2): Linear(in_features=256, out_features=256, bias=True)
        (3): ReLU()
        (4): Linear(in_features=256, out_features=4, bias=True)
      )
      (1): Sequential(
        (0): Linear(in_features=256, out_features=256, bias=True)
        (1): ReLU()
        (2): Linear(in_features=256, out_features=256, bias=True)
        (3): ReLU()
        (4): Linear(in_features=256, out_features=4, bias=True)
      )
      (2): Sequential(
        (0): Linear(in_features=256, out_features=256, bias=True)
        (1): ReLU()
        (2): Linear(in_features=256, out_features=256, bias=True)
        (3): ReLU()
        (4): Linear(in_features=256, out_features=4, bias=True)
      )
      (3): Sequential(
        (0): Linear(in_features=256, out_features=256, bias=True)
        (1): ReLU()
        (2): Linear(in_features=256, out_features=256, bias=True)
        (3): ReLU()
        (4): Linear(in_features=256, out_features=4, bias=True)
      )
      (4): Sequential(
        (0): Linear(in_features=256, out_features=256, bias=True)
        (1): ReLU()
        (2): Linear(in_features=256, out_features=256, bias=True)
        (3): ReLU()
        (4): Linear(in_features=256, out_features=4, bias=True)
      )
      (5): Sequential(
        (0): Linear(in_features=256, out_features=256, bias=True)
        (1): ReLU()
        (2): Linear(in_features=256, out_features=256, bias=True)
        (3): ReLU()
        (4): Linear(in_features=256, out_features=4, bias=True)
      )
    )
    (query_embedding): Embedding(300, 512)
  )
)

Process finished with exit code 0
