/home/zhangjp/miniconda3/envs/pth18/bin/python /home/zhangjp/project/xmdet216/tools/misc/check_model_arch.py

输入参数: Namespace(checkpoint=None, config='../../configs/xyz_rcnn/xyz_rcnn_swint_xpn_hlkt.py', device='cuda:1', score_thr=0.3)


########### Model Configs #################


########### Model Architecture #################

/home/zhangjp/miniconda3/envs/pth18/lib/python3.7/site-packages/mmcv/utils/misc.py:324: UserWarning: "dropout" is deprecated in `FFN.__init__`, please use "ffn_drop" instead
  f'"{src_arg_name}" is deprecated in '
XyzRCNN(
  (backbone): SwinTransformer(
    (patch_embed): PatchEmbed(
      (adap_padding): AdaptivePadding()
      (projection): Conv2d(3, 96, kernel_size=(4, 4), stride=(4, 4))
      (norm): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
    )
    (drop_after_pos): Dropout(p=0.0, inplace=False)
    (stages): ModuleList(
      (0): SwinBlockSequence(
        (blocks): ModuleList(
          (0): SwinBlock(
            (norm1): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
            (attn): ShiftWindowMSA(
              (w_msa): WindowMSA(
                (qkv): Linear(in_features=96, out_features=288, bias=True)
                (attn_drop): Dropout(p=0.0, inplace=False)
                (proj): Linear(in_features=96, out_features=96, bias=True)
                (proj_drop): Dropout(p=0.0, inplace=False)
                (softmax): Softmax(dim=-1)
              )
              (drop): DropPath()
            )
            (norm2): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
            (ffn): FFN(
              (activate): GELU()
              (layers): Sequential(
                (0): Sequential(
                  (0): Linear(in_features=96, out_features=384, bias=True)
                  (1): GELU()
                  (2): Dropout(p=0.0, inplace=False)
                )
                (1): Linear(in_features=384, out_features=96, bias=True)
                (2): Dropout(p=0.0, inplace=False)
              )
              (dropout_layer): DropPath()
            )
          )
          (1): SwinBlock(
            (norm1): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
            (attn): ShiftWindowMSA(
              (w_msa): WindowMSA(
                (qkv): Linear(in_features=96, out_features=288, bias=True)
                (attn_drop): Dropout(p=0.0, inplace=False)
                (proj): Linear(in_features=96, out_features=96, bias=True)
                (proj_drop): Dropout(p=0.0, inplace=False)
                (softmax): Softmax(dim=-1)
              )
              (drop): DropPath()
            )
            (norm2): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
            (ffn): FFN(
              (activate): GELU()
              (layers): Sequential(
                (0): Sequential(
                  (0): Linear(in_features=96, out_features=384, bias=True)
                  (1): GELU()
                  (2): Dropout(p=0.0, inplace=False)
                )
                (1): Linear(in_features=384, out_features=96, bias=True)
                (2): Dropout(p=0.0, inplace=False)
              )
              (dropout_layer): DropPath()
            )
          )
        )
        (downsample): PatchMerging(
          (adap_padding): AdaptivePadding()
          (sampler): Unfold(kernel_size=(2, 2), dilation=(1, 1), padding=(0, 0), stride=(2, 2))
          (norm): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
          (reduction): Linear(in_features=384, out_features=192, bias=False)
        )
      )
      (1): SwinBlockSequence(
        (blocks): ModuleList(
          (0): SwinBlock(
            (norm1): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
            (attn): ShiftWindowMSA(
              (w_msa): WindowMSA(
                (qkv): Linear(in_features=192, out_features=576, bias=True)
                (attn_drop): Dropout(p=0.0, inplace=False)
                (proj): Linear(in_features=192, out_features=192, bias=True)
                (proj_drop): Dropout(p=0.0, inplace=False)
                (softmax): Softmax(dim=-1)
              )
              (drop): DropPath()
            )
            (norm2): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
            (ffn): FFN(
              (activate): GELU()
              (layers): Sequential(
                (0): Sequential(
                  (0): Linear(in_features=192, out_features=768, bias=True)
                  (1): GELU()
                  (2): Dropout(p=0.0, inplace=False)
                )
                (1): Linear(in_features=768, out_features=192, bias=True)
                (2): Dropout(p=0.0, inplace=False)
              )
              (dropout_layer): DropPath()
            )
          )
          (1): SwinBlock(
            (norm1): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
            (attn): ShiftWindowMSA(
              (w_msa): WindowMSA(
                (qkv): Linear(in_features=192, out_features=576, bias=True)
                (attn_drop): Dropout(p=0.0, inplace=False)
                (proj): Linear(in_features=192, out_features=192, bias=True)
                (proj_drop): Dropout(p=0.0, inplace=False)
                (softmax): Softmax(dim=-1)
              )
              (drop): DropPath()
            )
            (norm2): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
            (ffn): FFN(
              (activate): GELU()
              (layers): Sequential(
                (0): Sequential(
                  (0): Linear(in_features=192, out_features=768, bias=True)
                  (1): GELU()
                  (2): Dropout(p=0.0, inplace=False)
                )
                (1): Linear(in_features=768, out_features=192, bias=True)
                (2): Dropout(p=0.0, inplace=False)
              )
              (dropout_layer): DropPath()
            )
          )
        )
        (downsample): PatchMerging(
          (adap_padding): AdaptivePadding()
          (sampler): Unfold(kernel_size=(2, 2), dilation=(1, 1), padding=(0, 0), stride=(2, 2))
          (norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (reduction): Linear(in_features=768, out_features=384, bias=False)
        )
      )
      (2): SwinBlockSequence(
        (blocks): ModuleList(
          (0): SwinBlock(
            (norm1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (attn): ShiftWindowMSA(
              (w_msa): WindowMSA(
                (qkv): Linear(in_features=384, out_features=1152, bias=True)
                (attn_drop): Dropout(p=0.0, inplace=False)
                (proj): Linear(in_features=384, out_features=384, bias=True)
                (proj_drop): Dropout(p=0.0, inplace=False)
                (softmax): Softmax(dim=-1)
              )
              (drop): DropPath()
            )
            (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (ffn): FFN(
              (activate): GELU()
              (layers): Sequential(
                (0): Sequential(
                  (0): Linear(in_features=384, out_features=1536, bias=True)
                  (1): GELU()
                  (2): Dropout(p=0.0, inplace=False)
                )
                (1): Linear(in_features=1536, out_features=384, bias=True)
                (2): Dropout(p=0.0, inplace=False)
              )
              (dropout_layer): DropPath()
            )
          )
          (1): SwinBlock(
            (norm1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (attn): ShiftWindowMSA(
              (w_msa): WindowMSA(
                (qkv): Linear(in_features=384, out_features=1152, bias=True)
                (attn_drop): Dropout(p=0.0, inplace=False)
                (proj): Linear(in_features=384, out_features=384, bias=True)
                (proj_drop): Dropout(p=0.0, inplace=False)
                (softmax): Softmax(dim=-1)
              )
              (drop): DropPath()
            )
            (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (ffn): FFN(
              (activate): GELU()
              (layers): Sequential(
                (0): Sequential(
                  (0): Linear(in_features=384, out_features=1536, bias=True)
                  (1): GELU()
                  (2): Dropout(p=0.0, inplace=False)
                )
                (1): Linear(in_features=1536, out_features=384, bias=True)
                (2): Dropout(p=0.0, inplace=False)
              )
              (dropout_layer): DropPath()
            )
          )
          (2): SwinBlock(
            (norm1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (attn): ShiftWindowMSA(
              (w_msa): WindowMSA(
                (qkv): Linear(in_features=384, out_features=1152, bias=True)
                (attn_drop): Dropout(p=0.0, inplace=False)
                (proj): Linear(in_features=384, out_features=384, bias=True)
                (proj_drop): Dropout(p=0.0, inplace=False)
                (softmax): Softmax(dim=-1)
              )
              (drop): DropPath()
            )
            (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (ffn): FFN(
              (activate): GELU()
              (layers): Sequential(
                (0): Sequential(
                  (0): Linear(in_features=384, out_features=1536, bias=True)
                  (1): GELU()
                  (2): Dropout(p=0.0, inplace=False)
                )
                (1): Linear(in_features=1536, out_features=384, bias=True)
                (2): Dropout(p=0.0, inplace=False)
              )
              (dropout_layer): DropPath()
            )
          )
          (3): SwinBlock(
            (norm1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (attn): ShiftWindowMSA(
              (w_msa): WindowMSA(
                (qkv): Linear(in_features=384, out_features=1152, bias=True)
                (attn_drop): Dropout(p=0.0, inplace=False)
                (proj): Linear(in_features=384, out_features=384, bias=True)
                (proj_drop): Dropout(p=0.0, inplace=False)
                (softmax): Softmax(dim=-1)
              )
              (drop): DropPath()
            )
            (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (ffn): FFN(
              (activate): GELU()
              (layers): Sequential(
                (0): Sequential(
                  (0): Linear(in_features=384, out_features=1536, bias=True)
                  (1): GELU()
                  (2): Dropout(p=0.0, inplace=False)
                )
                (1): Linear(in_features=1536, out_features=384, bias=True)
                (2): Dropout(p=0.0, inplace=False)
              )
              (dropout_layer): DropPath()
            )
          )
          (4): SwinBlock(
            (norm1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (attn): ShiftWindowMSA(
              (w_msa): WindowMSA(
                (qkv): Linear(in_features=384, out_features=1152, bias=True)
                (attn_drop): Dropout(p=0.0, inplace=False)
                (proj): Linear(in_features=384, out_features=384, bias=True)
                (proj_drop): Dropout(p=0.0, inplace=False)
                (softmax): Softmax(dim=-1)
              )
              (drop): DropPath()
            )
            (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (ffn): FFN(
              (activate): GELU()
              (layers): Sequential(
                (0): Sequential(
                  (0): Linear(in_features=384, out_features=1536, bias=True)
                  (1): GELU()
                  (2): Dropout(p=0.0, inplace=False)
                )
                (1): Linear(in_features=1536, out_features=384, bias=True)
                (2): Dropout(p=0.0, inplace=False)
              )
              (dropout_layer): DropPath()
            )
          )
          (5): SwinBlock(
            (norm1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (attn): ShiftWindowMSA(
              (w_msa): WindowMSA(
                (qkv): Linear(in_features=384, out_features=1152, bias=True)
                (attn_drop): Dropout(p=0.0, inplace=False)
                (proj): Linear(in_features=384, out_features=384, bias=True)
                (proj_drop): Dropout(p=0.0, inplace=False)
                (softmax): Softmax(dim=-1)
              )
              (drop): DropPath()
            )
            (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (ffn): FFN(
              (activate): GELU()
              (layers): Sequential(
                (0): Sequential(
                  (0): Linear(in_features=384, out_features=1536, bias=True)
                  (1): GELU()
                  (2): Dropout(p=0.0, inplace=False)
                )
                (1): Linear(in_features=1536, out_features=384, bias=True)
                (2): Dropout(p=0.0, inplace=False)
              )
              (dropout_layer): DropPath()
            )
          )
        )
        (downsample): PatchMerging(
          (adap_padding): AdaptivePadding()
          (sampler): Unfold(kernel_size=(2, 2), dilation=(1, 1), padding=(0, 0), stride=(2, 2))
          (norm): LayerNorm((1536,), eps=1e-05, elementwise_affine=True)
          (reduction): Linear(in_features=1536, out_features=768, bias=False)
        )
      )
      (3): SwinBlockSequence(
        (blocks): ModuleList(
          (0): SwinBlock(
            (norm1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (attn): ShiftWindowMSA(
              (w_msa): WindowMSA(
                (qkv): Linear(in_features=768, out_features=2304, bias=True)
                (attn_drop): Dropout(p=0.0, inplace=False)
                (proj): Linear(in_features=768, out_features=768, bias=True)
                (proj_drop): Dropout(p=0.0, inplace=False)
                (softmax): Softmax(dim=-1)
              )
              (drop): DropPath()
            )
            (norm2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (ffn): FFN(
              (activate): GELU()
              (layers): Sequential(
                (0): Sequential(
                  (0): Linear(in_features=768, out_features=3072, bias=True)
                  (1): GELU()
                  (2): Dropout(p=0.0, inplace=False)
                )
                (1): Linear(in_features=3072, out_features=768, bias=True)
                (2): Dropout(p=0.0, inplace=False)
              )
              (dropout_layer): DropPath()
            )
          )
          (1): SwinBlock(
            (norm1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (attn): ShiftWindowMSA(
              (w_msa): WindowMSA(
                (qkv): Linear(in_features=768, out_features=2304, bias=True)
                (attn_drop): Dropout(p=0.0, inplace=False)
                (proj): Linear(in_features=768, out_features=768, bias=True)
                (proj_drop): Dropout(p=0.0, inplace=False)
                (softmax): Softmax(dim=-1)
              )
              (drop): DropPath()
            )
            (norm2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (ffn): FFN(
              (activate): GELU()
              (layers): Sequential(
                (0): Sequential(
                  (0): Linear(in_features=768, out_features=3072, bias=True)
                  (1): GELU()
                  (2): Dropout(p=0.0, inplace=False)
                )
                (1): Linear(in_features=3072, out_features=768, bias=True)
                (2): Dropout(p=0.0, inplace=False)
              )
              (dropout_layer): DropPath()
            )
          )
        )
      )
    )
    (norm0): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
    (norm1): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
    (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
    (norm3): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
  )
  init_cfg={'type': 'Pretrained', 'checkpoint': '/home/zhangjp/softlink/Pretrained/swin_tiny_224_b16x64_300e_imagenet_20210616_090925-66df6be6.pth'}
  (neck): XPN(
    (lateral_convs): ModuleList(
      (0): ConvModule(
        (conv): Conv2d(96, 256, kernel_size=(1, 1), stride=(1, 1))
      )
      (1): ConvModule(
        (conv): Conv2d(192, 256, kernel_size=(1, 1), stride=(1, 1))
      )
      (2): ConvModule(
        (conv): Conv2d(384, 256, kernel_size=(1, 1), stride=(1, 1))
      )
      (3): ConvModule(
        (conv): Conv2d(768, 256, kernel_size=(1, 1), stride=(1, 1))
      )
    )
    (fpn_convs): ModuleList(
      (0): ConvModule(
        (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (1): ConvModule(
        (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (2): ConvModule(
        (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (3): ConvModule(
        (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
    )
  )
  init_cfg={'type': 'Xavier', 'layer': 'Conv2d', 'distribution': 'uniform'}
  (rpn_head): XyzRPNHead(
    (init_proposal_bboxes): Embedding(100, 4)
    (init_proposal_features): Embedding(100, 256)
  )
  (roi_head): XyzRoIHead(
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
      (0): XYZHead(
        (loss_cls): FocalLoss()
        (loss_bbox): L1Loss()
        (fc_cls): Linear(in_features=256, out_features=10, bias=True)
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
      (1): XYZHead(
        (loss_cls): FocalLoss()
        (loss_bbox): L1Loss()
        (fc_cls): Linear(in_features=256, out_features=10, bias=True)
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
      (2): XYZHead(
        (loss_cls): FocalLoss()
        (loss_bbox): L1Loss()
        (fc_cls): Linear(in_features=256, out_features=10, bias=True)
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
      (3): XYZHead(
        (loss_cls): FocalLoss()
        (loss_bbox): L1Loss()
        (fc_cls): Linear(in_features=256, out_features=10, bias=True)
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
      (4): XYZHead(
        (loss_cls): FocalLoss()
        (loss_bbox): L1Loss()
        (fc_cls): Linear(in_features=256, out_features=10, bias=True)
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
      (5): XYZHead(
        (loss_cls): FocalLoss()
        (loss_bbox): L1Loss()
        (fc_cls): Linear(in_features=256, out_features=10, bias=True)
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

Process finished with exit code 0
