============== model arch start ==============
SparseRCNN(
  (backbone): ResNet(
    (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu): ReLU(inplace=True)
    (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    (layer1): ResLayer(
      (0): BasicBlock(
        (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
      (1): BasicBlock(
        (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
    )
  )
  init_cfg={'type': 'Pretrained', 'checkpoint': '/home/softlink/Pretrained/resnet18-5c106cde.pth'}
  (neck): FPN(
    (lateral_convs): ModuleList(
      (0): ConvModule(
        (conv): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1))
        (activate): ReLU()
      )
      (1): ConvModule(
        (conv): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1))
        (activate): ReLU()
      )
      (2): ConvModule(
        (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        (activate): ReLU()
      )
      (3): ConvModule(
        (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
        (activate): ReLU()
      )
    )
    (fpn_convs): ModuleList(
      (0): ConvModule(
        (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (activate): ReLU()
      )
      (1): ConvModule(
        (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (activate): ReLU()
      )
      (2): ConvModule(
        (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (activate): ReLU()
      )
      (3): ConvModule(
        (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (activate): ReLU()
      )
    )
  )
  init_cfg={'type': 'Xavier', 'layer': 'Conv2d', 'distribution': 'uniform'}
  (rpn_head): EmbeddingRPNHead(
    (init_proposal_bboxes): Embedding(100, 4)
    (init_proposal_features): Embedding(100, 256)
  )
  (roi_head): SparseRoIHead(
    (bbox_roi_extractor): ModuleList(
      (0): SingleRoIExtractor(
        (roi_layers): ModuleList(
          (0): RoIAlign(output_size=(7, 7), spatial_scale=0.25, sampling_ratio=2, pool_mode=avg, aligned=True, use_torchvision=False)
          (1): RoIAlign(output_size=(7, 7), spatial_scale=0.125, sampling_ratio=2, pool_mode=avg, aligned=True, use_torchvision=False)
          (2): RoIAlign(output_size=(7, 7), spatial_scale=0.0625, sampling_ratio=2, pool_mode=avg, aligned=True, use_torchvision=False)
          (3): RoIAlign(output_size=(7, 7), spatial_scale=0.03125, sampling_ratio=2, pool_mode=avg, aligned=True, use_torchvision=False)
        )
      )
      (1): SingleRoIExtractor(
        (roi_layers): ModuleList(
          (0): RoIAlign(output_size=(7, 7), spatial_scale=0.25, sampling_ratio=2, pool_mode=avg, aligned=True, use_torchvision=False)
          (1): RoIAlign(output_size=(7, 7), spatial_scale=0.125, sampling_ratio=2, pool_mode=avg, aligned=True, use_torchvision=False)
          (2): RoIAlign(output_size=(7, 7), spatial_scale=0.0625, sampling_ratio=2, pool_mode=avg, aligned=True, use_torchvision=False)
          (3): RoIAlign(output_size=(7, 7), spatial_scale=0.03125, sampling_ratio=2, pool_mode=avg, aligned=True, use_torchvision=False)
        )
      )
      (2): SingleRoIExtractor(
        (roi_layers): ModuleList(
          (0): RoIAlign(output_size=(7, 7), spatial_scale=0.25, sampling_ratio=2, pool_mode=avg, aligned=True, use_torchvision=False)
          (1): RoIAlign(output_size=(7, 7), spatial_scale=0.125, sampling_ratio=2, pool_mode=avg, aligned=True, use_torchvision=False)
          (2): RoIAlign(output_size=(7, 7), spatial_scale=0.0625, sampling_ratio=2, pool_mode=avg, aligned=True, use_torchvision=False)
          (3): RoIAlign(output_size=(7, 7), spatial_scale=0.03125, sampling_ratio=2, pool_mode=avg, aligned=True, use_torchvision=False)
        )
      )
      (3): SingleRoIExtractor(
        (roi_layers): ModuleList(
          (0): RoIAlign(output_size=(7, 7), spatial_scale=0.25, sampling_ratio=2, pool_mode=avg, aligned=True, use_torchvision=False)
          (1): RoIAlign(output_size=(7, 7), spatial_scale=0.125, sampling_ratio=2, pool_mode=avg, aligned=True, use_torchvision=False)
          (2): RoIAlign(output_size=(7, 7), spatial_scale=0.0625, sampling_ratio=2, pool_mode=avg, aligned=True, use_torchvision=False)
          (3): RoIAlign(output_size=(7, 7), spatial_scale=0.03125, sampling_ratio=2, pool_mode=avg, aligned=True, use_torchvision=False)
        )
      )
      (4): SingleRoIExtractor(
        (roi_layers): ModuleList(
          (0): RoIAlign(output_size=(7, 7), spatial_scale=0.25, sampling_ratio=2, pool_mode=avg, aligned=True, use_torchvision=False)
          (1): RoIAlign(output_size=(7, 7), spatial_scale=0.125, sampling_ratio=2, pool_mode=avg, aligned=True, use_torchvision=False)
          (2): RoIAlign(output_size=(7, 7), spatial_scale=0.0625, sampling_ratio=2, pool_mode=avg, aligned=True, use_torchvision=False)
          (3): RoIAlign(output_size=(7, 7), spatial_scale=0.03125, sampling_ratio=2, pool_mode=avg, aligned=True, use_torchvision=False)
        )
      )
      (5): SingleRoIExtractor(
        (roi_layers): ModuleList(
          (0): RoIAlign(output_size=(7, 7), spatial_scale=0.25, sampling_ratio=2, pool_mode=avg, aligned=True, use_torchvision=False)
          (1): RoIAlign(output_size=(7, 7), spatial_scale=0.125, sampling_ratio=2, pool_mode=avg, aligned=True, use_torchvision=False)
          (2): RoIAlign(output_size=(7, 7), spatial_scale=0.0625, sampling_ratio=2, pool_mode=avg, aligned=True, use_torchvision=False)
          (3): RoIAlign(output_size=(7, 7), spatial_scale=0.03125, sampling_ratio=2, pool_mode=avg, aligned=True, use_torchvision=False)
        )
      )
    )
    (bbox_head): ModuleList(
      (0): DIIHead(
        (loss_cls): FocalLoss()
        (loss_bbox): L1Loss()
        (fc_cls): Linear(in_features=256, out_features=80, bias=True)
        (fc_reg): Linear(in_features=256, out_features=4, bias=True)
        (loss_iou): GIoULoss()
        (attention): MultiheadAttention(
          (attn): MultiheadAttention(
            (out_proj): _LinearWithBias(in_features=256, out_features=256, bias=True)
          )
          (proj_drop): Dropout(p=0.0, inplace=False)
          (dropout_layer): Dropout(p=0.0, inplace=False)
        )
        (attention_norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        (instance_interactive_conv): DynamicConv(
          (dynamic_layer): Linear(in_features=256, out_features=32768, bias=True)
          (norm_in): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
          (norm_out): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (activation): ReLU(inplace=True)
          (fc_layer): Linear(in_features=12544, out_features=256, bias=True)
          (fc_norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        )
        (instance_interactive_conv_dropout): Dropout(p=0.0, inplace=False)
        (instance_interactive_conv_norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        (ffn): FFN(
          (activate): ReLU(inplace=True)
          (layers): Sequential(
            (0): Sequential(
              (0): Linear(in_features=256, out_features=2048, bias=True)
              (1): ReLU(inplace=True)
              (2): Dropout(p=0.0, inplace=False)
            )
            (1): Linear(in_features=2048, out_features=256, bias=True)
            (2): Dropout(p=0.0, inplace=False)
          )
          (dropout_layer): Identity()
        )
        (ffn_norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        (cls_fcs): ModuleList(
          (0): Linear(in_features=256, out_features=256, bias=False)
          (1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (2): ReLU(inplace=True)
        )
        (reg_fcs): ModuleList(
          (0): Linear(in_features=256, out_features=256, bias=False)
          (1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (2): ReLU(inplace=True)
          (3): Linear(in_features=256, out_features=256, bias=False)
          (4): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (5): ReLU(inplace=True)
          (6): Linear(in_features=256, out_features=256, bias=False)
          (7): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (8): ReLU(inplace=True)
        )
      )
      init_cfg=[{'type': 'Normal', 'std': 0.01, 'override': {'name': 'fc_cls'}}, {'type': 'Normal', 'std': 0.001, 'override': {'name': 'fc_reg'}}]
      (1): DIIHead(
        (loss_cls): FocalLoss()
        (loss_bbox): L1Loss()
        (fc_cls): Linear(in_features=256, out_features=80, bias=True)
        (fc_reg): Linear(in_features=256, out_features=4, bias=True)
        (loss_iou): GIoULoss()
        (attention): MultiheadAttention(
          (attn): MultiheadAttention(
            (out_proj): _LinearWithBias(in_features=256, out_features=256, bias=True)
          )
          (proj_drop): Dropout(p=0.0, inplace=False)
          (dropout_layer): Dropout(p=0.0, inplace=False)
        )
        (attention_norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        (instance_interactive_conv): DynamicConv(
          (dynamic_layer): Linear(in_features=256, out_features=32768, bias=True)
          (norm_in): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
          (norm_out): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (activation): ReLU(inplace=True)
          (fc_layer): Linear(in_features=12544, out_features=256, bias=True)
          (fc_norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        )
        (instance_interactive_conv_dropout): Dropout(p=0.0, inplace=False)
        (instance_interactive_conv_norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        (ffn): FFN(
          (activate): ReLU(inplace=True)
          (layers): Sequential(
            (0): Sequential(
              (0): Linear(in_features=256, out_features=2048, bias=True)
              (1): ReLU(inplace=True)
              (2): Dropout(p=0.0, inplace=False)
            )
            (1): Linear(in_features=2048, out_features=256, bias=True)
            (2): Dropout(p=0.0, inplace=False)
          )
          (dropout_layer): Identity()
        )
        (ffn_norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        (cls_fcs): ModuleList(
          (0): Linear(in_features=256, out_features=256, bias=False)
          (1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (2): ReLU(inplace=True)
        )
        (reg_fcs): ModuleList(
          (0): Linear(in_features=256, out_features=256, bias=False)
          (1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (2): ReLU(inplace=True)
          (3): Linear(in_features=256, out_features=256, bias=False)
          (4): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (5): ReLU(inplace=True)
          (6): Linear(in_features=256, out_features=256, bias=False)
          (7): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (8): ReLU(inplace=True)
        )
      )
      init_cfg=[{'type': 'Normal', 'std': 0.01, 'override': {'name': 'fc_cls'}}, {'type': 'Normal', 'std': 0.001, 'override': {'name': 'fc_reg'}}]
      (2): DIIHead(
        (loss_cls): FocalLoss()
        (loss_bbox): L1Loss()
        (fc_cls): Linear(in_features=256, out_features=80, bias=True)
        (fc_reg): Linear(in_features=256, out_features=4, bias=True)
        (loss_iou): GIoULoss()
        (attention): MultiheadAttention(
          (attn): MultiheadAttention(
            (out_proj): _LinearWithBias(in_features=256, out_features=256, bias=True)
          )
          (proj_drop): Dropout(p=0.0, inplace=False)
          (dropout_layer): Dropout(p=0.0, inplace=False)
        )
        (attention_norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        (instance_interactive_conv): DynamicConv(
          (dynamic_layer): Linear(in_features=256, out_features=32768, bias=True)
          (norm_in): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
          (norm_out): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (activation): ReLU(inplace=True)
          (fc_layer): Linear(in_features=12544, out_features=256, bias=True)
          (fc_norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        )
        (instance_interactive_conv_dropout): Dropout(p=0.0, inplace=False)
        (instance_interactive_conv_norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        (ffn): FFN(
          (activate): ReLU(inplace=True)
          (layers): Sequential(
            (0): Sequential(
              (0): Linear(in_features=256, out_features=2048, bias=True)
              (1): ReLU(inplace=True)
              (2): Dropout(p=0.0, inplace=False)
            )
            (1): Linear(in_features=2048, out_features=256, bias=True)
            (2): Dropout(p=0.0, inplace=False)
          )
          (dropout_layer): Identity()
        )
        (ffn_norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        (cls_fcs): ModuleList(
          (0): Linear(in_features=256, out_features=256, bias=False)
          (1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (2): ReLU(inplace=True)
        )
        (reg_fcs): ModuleList(
          (0): Linear(in_features=256, out_features=256, bias=False)
          (1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (2): ReLU(inplace=True)
          (3): Linear(in_features=256, out_features=256, bias=False)
          (4): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (5): ReLU(inplace=True)
          (6): Linear(in_features=256, out_features=256, bias=False)
          (7): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (8): ReLU(inplace=True)
        )
      )
      init_cfg=[{'type': 'Normal', 'std': 0.01, 'override': {'name': 'fc_cls'}}, {'type': 'Normal', 'std': 0.001, 'override': {'name': 'fc_reg'}}]
      (3): DIIHead(
        (loss_cls): FocalLoss()
        (loss_bbox): L1Loss()
        (fc_cls): Linear(in_features=256, out_features=80, bias=True)
        (fc_reg): Linear(in_features=256, out_features=4, bias=True)
        (loss_iou): GIoULoss()
        (attention): MultiheadAttention(
          (attn): MultiheadAttention(
            (out_proj): _LinearWithBias(in_features=256, out_features=256, bias=True)
          )
          (proj_drop): Dropout(p=0.0, inplace=False)
          (dropout_layer): Dropout(p=0.0, inplace=False)
        )
        (attention_norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        (instance_interactive_conv): DynamicConv(
          (dynamic_layer): Linear(in_features=256, out_features=32768, bias=True)
          (norm_in): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
          (norm_out): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (activation): ReLU(inplace=True)
          (fc_layer): Linear(in_features=12544, out_features=256, bias=True)
          (fc_norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        )
        (instance_interactive_conv_dropout): Dropout(p=0.0, inplace=False)
        (instance_interactive_conv_norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        (ffn): FFN(
          (activate): ReLU(inplace=True)
          (layers): Sequential(
            (0): Sequential(
              (0): Linear(in_features=256, out_features=2048, bias=True)
              (1): ReLU(inplace=True)
              (2): Dropout(p=0.0, inplace=False)
            )
            (1): Linear(in_features=2048, out_features=256, bias=True)
            (2): Dropout(p=0.0, inplace=False)
          )
          (dropout_layer): Identity()
        )
        (ffn_norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        (cls_fcs): ModuleList(
          (0): Linear(in_features=256, out_features=256, bias=False)
          (1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (2): ReLU(inplace=True)
        )
        (reg_fcs): ModuleList(
          (0): Linear(in_features=256, out_features=256, bias=False)
          (1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (2): ReLU(inplace=True)
          (3): Linear(in_features=256, out_features=256, bias=False)
          (4): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (5): ReLU(inplace=True)
          (6): Linear(in_features=256, out_features=256, bias=False)
          (7): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (8): ReLU(inplace=True)
        )
      )
      init_cfg=[{'type': 'Normal', 'std': 0.01, 'override': {'name': 'fc_cls'}}, {'type': 'Normal', 'std': 0.001, 'override': {'name': 'fc_reg'}}]
      (4): DIIHead(
        (loss_cls): FocalLoss()
        (loss_bbox): L1Loss()
        (fc_cls): Linear(in_features=256, out_features=80, bias=True)
        (fc_reg): Linear(in_features=256, out_features=4, bias=True)
        (loss_iou): GIoULoss()
        (attention): MultiheadAttention(
          (attn): MultiheadAttention(
            (out_proj): _LinearWithBias(in_features=256, out_features=256, bias=True)
          )
          (proj_drop): Dropout(p=0.0, inplace=False)
          (dropout_layer): Dropout(p=0.0, inplace=False)
        )
        (attention_norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        (instance_interactive_conv): DynamicConv(
          (dynamic_layer): Linear(in_features=256, out_features=32768, bias=True)
          (norm_in): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
          (norm_out): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (activation): ReLU(inplace=True)
          (fc_layer): Linear(in_features=12544, out_features=256, bias=True)
          (fc_norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        )
        (instance_interactive_conv_dropout): Dropout(p=0.0, inplace=False)
        (instance_interactive_conv_norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        (ffn): FFN(
          (activate): ReLU(inplace=True)
          (layers): Sequential(
            (0): Sequential(
              (0): Linear(in_features=256, out_features=2048, bias=True)
              (1): ReLU(inplace=True)
              (2): Dropout(p=0.0, inplace=False)
            )
            (1): Linear(in_features=2048, out_features=256, bias=True)
            (2): Dropout(p=0.0, inplace=False)
          )
          (dropout_layer): Identity()
        )
        (ffn_norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        (cls_fcs): ModuleList(
          (0): Linear(in_features=256, out_features=256, bias=False)
          (1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (2): ReLU(inplace=True)
        )
        (reg_fcs): ModuleList(
          (0): Linear(in_features=256, out_features=256, bias=False)
          (1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (2): ReLU(inplace=True)
          (3): Linear(in_features=256, out_features=256, bias=False)
          (4): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (5): ReLU(inplace=True)
          (6): Linear(in_features=256, out_features=256, bias=False)
          (7): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (8): ReLU(inplace=True)
        )
      )
      init_cfg=[{'type': 'Normal', 'std': 0.01, 'override': {'name': 'fc_cls'}}, {'type': 'Normal', 'std': 0.001, 'override': {'name': 'fc_reg'}}]
      (5): DIIHead(
        (loss_cls): FocalLoss()
        (loss_bbox): L1Loss()
        (fc_cls): Linear(in_features=256, out_features=80, bias=True)
        (fc_reg): Linear(in_features=256, out_features=4, bias=True)
        (loss_iou): GIoULoss()
        (attention): MultiheadAttention(
          (attn): MultiheadAttention(
            (out_proj): _LinearWithBias(in_features=256, out_features=256, bias=True)
          )
          (proj_drop): Dropout(p=0.0, inplace=False)
          (dropout_layer): Dropout(p=0.0, inplace=False)
        )
        (attention_norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        (instance_interactive_conv): DynamicConv(
          (dynamic_layer): Linear(in_features=256, out_features=32768, bias=True)
          (norm_in): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
          (norm_out): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (activation): ReLU(inplace=True)
          (fc_layer): Linear(in_features=12544, out_features=256, bias=True)
          (fc_norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        )
        (instance_interactive_conv_dropout): Dropout(p=0.0, inplace=False)
        (instance_interactive_conv_norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        (ffn): FFN(
          (activate): ReLU(inplace=True)
          (layers): Sequential(
            (0): Sequential(
              (0): Linear(in_features=256, out_features=2048, bias=True)
              (1): ReLU(inplace=True)
              (2): Dropout(p=0.0, inplace=False)
            )
            (1): Linear(in_features=2048, out_features=256, bias=True)
            (2): Dropout(p=0.0, inplace=False)
          )
          (dropout_layer): Identity()
        )
        (ffn_norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        (cls_fcs): ModuleList(
          (0): Linear(in_features=256, out_features=256, bias=False)
          (1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (2): ReLU(inplace=True)
        )
        (reg_fcs): ModuleList(
          (0): Linear(in_features=256, out_features=256, bias=False)
          (1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (2): ReLU(inplace=True)
          (3): Linear(in_features=256, out_features=256, bias=False)
          (4): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (5): ReLU(inplace=True)
          (6): Linear(in_features=256, out_features=256, bias=False)
          (7): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (8): ReLU(inplace=True)
        )
      )
      init_cfg=[{'type': 'Normal', 'std': 0.01, 'override': {'name': 'fc_cls'}}, {'type': 'Normal', 'std': 0.001, 'override': {'name': 'fc_reg'}}]
    )
  )
)
============== model arch end ==============