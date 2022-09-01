/home/zhangjp/miniconda3/envs/pth18/bin/python /home/zhangjp/project/xmdet216/tools/misc/check_model_arch.py

输入参数: Namespace(checkpoint=None, config='../../configs/xyz_rcnn/xyz_rcnn_pvtb2_xpn_hlkt.py', device='cuda:1', score_thr=0.3)


########### Model Architecture #################

/home/zhangjp/miniconda3/envs/pth18/lib/python3.7/site-packages/mmcv/utils/misc.py:324: 
UserWarning: "dropout" is deprecated in `FFN.__init__`, please use "ffn_drop" instead
  f'"{src_arg_name}" is deprecated in '
XyzRCNN(
  (backbone): PyramidVisionTransformerV2(
    (layers): ModuleList(
      (0): ModuleList(
        (0): PatchEmbed(
          (projection): Conv2d(3, 64, kernel_size=(7, 7), stride=(4, 4), padding=(3, 3))
          (norm): LayerNorm((64,), eps=1e-06, elementwise_affine=True)
        )
        (1): ModuleList(
          (0): PVTEncoderLayer(
            (norm1): LayerNorm((64,), eps=1e-06, elementwise_affine=True)
            (attn): SpatialReductionAttention(
              (attn): MultiheadAttention(
                (out_proj): _LinearWithBias(in_features=64, out_features=64, bias=True)
              )
              (proj_drop): Dropout(p=0.0, inplace=False)
              (dropout_layer): DropPath()
              (sr): Conv2d(64, 64, kernel_size=(8, 8), stride=(8, 8))
              (norm): LayerNorm((64,), eps=1e-06, elementwise_affine=True)
            )
            (norm2): LayerNorm((64,), eps=1e-06, elementwise_affine=True)
            (ffn): MixFFN(
              (layers): Sequential(
                (0): Conv2d(64, 512, kernel_size=(1, 1), stride=(1, 1))
                (1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=512)
                (2): GELU()
                (3): Dropout(p=0.0, inplace=False)
                (4): Conv2d(512, 64, kernel_size=(1, 1), stride=(1, 1))
                (5): Dropout(p=0.0, inplace=False)
              )
              (dropout_layer): DropPath()
            )
          )
          (1): PVTEncoderLayer(
            (norm1): LayerNorm((64,), eps=1e-06, elementwise_affine=True)
            (attn): SpatialReductionAttention(
              (attn): MultiheadAttention(
                (out_proj): _LinearWithBias(in_features=64, out_features=64, bias=True)
              )
              (proj_drop): Dropout(p=0.0, inplace=False)
              (dropout_layer): DropPath()
              (sr): Conv2d(64, 64, kernel_size=(8, 8), stride=(8, 8))
              (norm): LayerNorm((64,), eps=1e-06, elementwise_affine=True)
            )
            (norm2): LayerNorm((64,), eps=1e-06, elementwise_affine=True)
            (ffn): MixFFN(
              (layers): Sequential(
                (0): Conv2d(64, 512, kernel_size=(1, 1), stride=(1, 1))
                (1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=512)
                (2): GELU()
                (3): Dropout(p=0.0, inplace=False)
                (4): Conv2d(512, 64, kernel_size=(1, 1), stride=(1, 1))
                (5): Dropout(p=0.0, inplace=False)
              )
              (dropout_layer): DropPath()
            )
          )
          (2): PVTEncoderLayer(
            (norm1): LayerNorm((64,), eps=1e-06, elementwise_affine=True)
            (attn): SpatialReductionAttention(
              (attn): MultiheadAttention(
                (out_proj): _LinearWithBias(in_features=64, out_features=64, bias=True)
              )
              (proj_drop): Dropout(p=0.0, inplace=False)
              (dropout_layer): DropPath()
              (sr): Conv2d(64, 64, kernel_size=(8, 8), stride=(8, 8))
              (norm): LayerNorm((64,), eps=1e-06, elementwise_affine=True)
            )
            (norm2): LayerNorm((64,), eps=1e-06, elementwise_affine=True)
            (ffn): MixFFN(
              (layers): Sequential(
                (0): Conv2d(64, 512, kernel_size=(1, 1), stride=(1, 1))
                (1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=512)
                (2): GELU()
                (3): Dropout(p=0.0, inplace=False)
                (4): Conv2d(512, 64, kernel_size=(1, 1), stride=(1, 1))
                (5): Dropout(p=0.0, inplace=False)
              )
              (dropout_layer): DropPath()
            )
          )
        )
        (2): LayerNorm((64,), eps=1e-06, elementwise_affine=True)
      )
      (1): ModuleList(
        (0): PatchEmbed(
          (projection): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
          (norm): LayerNorm((128,), eps=1e-06, elementwise_affine=True)
        )
        (1): ModuleList(
          (0): PVTEncoderLayer(
            (norm1): LayerNorm((128,), eps=1e-06, elementwise_affine=True)
            (attn): SpatialReductionAttention(
              (attn): MultiheadAttention(
                (out_proj): _LinearWithBias(in_features=128, out_features=128, bias=True)
              )
              (proj_drop): Dropout(p=0.0, inplace=False)
              (dropout_layer): DropPath()
              (sr): Conv2d(128, 128, kernel_size=(4, 4), stride=(4, 4))
              (norm): LayerNorm((128,), eps=1e-06, elementwise_affine=True)
            )
            (norm2): LayerNorm((128,), eps=1e-06, elementwise_affine=True)
            (ffn): MixFFN(
              (layers): Sequential(
                (0): Conv2d(128, 1024, kernel_size=(1, 1), stride=(1, 1))
                (1): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1024)
                (2): GELU()
                (3): Dropout(p=0.0, inplace=False)
                (4): Conv2d(1024, 128, kernel_size=(1, 1), stride=(1, 1))
                (5): Dropout(p=0.0, inplace=False)
              )
              (dropout_layer): DropPath()
            )
          )
          (1): PVTEncoderLayer(
            (norm1): LayerNorm((128,), eps=1e-06, elementwise_affine=True)
            (attn): SpatialReductionAttention(
              (attn): MultiheadAttention(
                (out_proj): _LinearWithBias(in_features=128, out_features=128, bias=True)
              )
              (proj_drop): Dropout(p=0.0, inplace=False)
              (dropout_layer): DropPath()
              (sr): Conv2d(128, 128, kernel_size=(4, 4), stride=(4, 4))
              (norm): LayerNorm((128,), eps=1e-06, elementwise_affine=True)
            )
            (norm2): LayerNorm((128,), eps=1e-06, elementwise_affine=True)
            (ffn): MixFFN(
              (layers): Sequential(
                (0): Conv2d(128, 1024, kernel_size=(1, 1), stride=(1, 1))
                (1): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1024)
                (2): GELU()
                (3): Dropout(p=0.0, inplace=False)
                (4): Conv2d(1024, 128, kernel_size=(1, 1), stride=(1, 1))
                (5): Dropout(p=0.0, inplace=False)
              )
              (dropout_layer): DropPath()
            )
          )
          (2): PVTEncoderLayer(
            (norm1): LayerNorm((128,), eps=1e-06, elementwise_affine=True)
            (attn): SpatialReductionAttention(
              (attn): MultiheadAttention(
                (out_proj): _LinearWithBias(in_features=128, out_features=128, bias=True)
              )
              (proj_drop): Dropout(p=0.0, inplace=False)
              (dropout_layer): DropPath()
              (sr): Conv2d(128, 128, kernel_size=(4, 4), stride=(4, 4))
              (norm): LayerNorm((128,), eps=1e-06, elementwise_affine=True)
            )
            (norm2): LayerNorm((128,), eps=1e-06, elementwise_affine=True)
            (ffn): MixFFN(
              (layers): Sequential(
                (0): Conv2d(128, 1024, kernel_size=(1, 1), stride=(1, 1))
                (1): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1024)
                (2): GELU()
                (3): Dropout(p=0.0, inplace=False)
                (4): Conv2d(1024, 128, kernel_size=(1, 1), stride=(1, 1))
                (5): Dropout(p=0.0, inplace=False)
              )
              (dropout_layer): DropPath()
            )
          )
          (3): PVTEncoderLayer(
            (norm1): LayerNorm((128,), eps=1e-06, elementwise_affine=True)
            (attn): SpatialReductionAttention(
              (attn): MultiheadAttention(
                (out_proj): _LinearWithBias(in_features=128, out_features=128, bias=True)
              )
              (proj_drop): Dropout(p=0.0, inplace=False)
              (dropout_layer): DropPath()
              (sr): Conv2d(128, 128, kernel_size=(4, 4), stride=(4, 4))
              (norm): LayerNorm((128,), eps=1e-06, elementwise_affine=True)
            )
            (norm2): LayerNorm((128,), eps=1e-06, elementwise_affine=True)
            (ffn): MixFFN(
              (layers): Sequential(
                (0): Conv2d(128, 1024, kernel_size=(1, 1), stride=(1, 1))
                (1): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1024)
                (2): GELU()
                (3): Dropout(p=0.0, inplace=False)
                (4): Conv2d(1024, 128, kernel_size=(1, 1), stride=(1, 1))
                (5): Dropout(p=0.0, inplace=False)
              )
              (dropout_layer): DropPath()
            )
          )
        )
        (2): LayerNorm((128,), eps=1e-06, elementwise_affine=True)
      )
      (2): ModuleList(
        (0): PatchEmbed(
          (projection): Conv2d(128, 320, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
          (norm): LayerNorm((320,), eps=1e-06, elementwise_affine=True)
        )
        (1): ModuleList(
          (0): PVTEncoderLayer(
            (norm1): LayerNorm((320,), eps=1e-06, elementwise_affine=True)
            (attn): SpatialReductionAttention(
              (attn): MultiheadAttention(
                (out_proj): _LinearWithBias(in_features=320, out_features=320, bias=True)
              )
              (proj_drop): Dropout(p=0.0, inplace=False)
              (dropout_layer): DropPath()
              (sr): Conv2d(320, 320, kernel_size=(2, 2), stride=(2, 2))
              (norm): LayerNorm((320,), eps=1e-06, elementwise_affine=True)
            )
            (norm2): LayerNorm((320,), eps=1e-06, elementwise_affine=True)
            (ffn): MixFFN(
              (layers): Sequential(
                (0): Conv2d(320, 1280, kernel_size=(1, 1), stride=(1, 1))
                (1): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1280)
                (2): GELU()
                (3): Dropout(p=0.0, inplace=False)
                (4): Conv2d(1280, 320, kernel_size=(1, 1), stride=(1, 1))
                (5): Dropout(p=0.0, inplace=False)
              )
              (dropout_layer): DropPath()
            )
          )
          (1): PVTEncoderLayer(
            (norm1): LayerNorm((320,), eps=1e-06, elementwise_affine=True)
            (attn): SpatialReductionAttention(
              (attn): MultiheadAttention(
                (out_proj): _LinearWithBias(in_features=320, out_features=320, bias=True)
              )
              (proj_drop): Dropout(p=0.0, inplace=False)
              (dropout_layer): DropPath()
              (sr): Conv2d(320, 320, kernel_size=(2, 2), stride=(2, 2))
              (norm): LayerNorm((320,), eps=1e-06, elementwise_affine=True)
            )
            (norm2): LayerNorm((320,), eps=1e-06, elementwise_affine=True)
            (ffn): MixFFN(
              (layers): Sequential(
                (0): Conv2d(320, 1280, kernel_size=(1, 1), stride=(1, 1))
                (1): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1280)
                (2): GELU()
                (3): Dropout(p=0.0, inplace=False)
                (4): Conv2d(1280, 320, kernel_size=(1, 1), stride=(1, 1))
                (5): Dropout(p=0.0, inplace=False)
              )
              (dropout_layer): DropPath()
            )
          )
          (2): PVTEncoderLayer(
            (norm1): LayerNorm((320,), eps=1e-06, elementwise_affine=True)
            (attn): SpatialReductionAttention(
              (attn): MultiheadAttention(
                (out_proj): _LinearWithBias(in_features=320, out_features=320, bias=True)
              )
              (proj_drop): Dropout(p=0.0, inplace=False)
              (dropout_layer): DropPath()
              (sr): Conv2d(320, 320, kernel_size=(2, 2), stride=(2, 2))
              (norm): LayerNorm((320,), eps=1e-06, elementwise_affine=True)
            )
            (norm2): LayerNorm((320,), eps=1e-06, elementwise_affine=True)
            (ffn): MixFFN(
              (layers): Sequential(
                (0): Conv2d(320, 1280, kernel_size=(1, 1), stride=(1, 1))
                (1): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1280)
                (2): GELU()
                (3): Dropout(p=0.0, inplace=False)
                (4): Conv2d(1280, 320, kernel_size=(1, 1), stride=(1, 1))
                (5): Dropout(p=0.0, inplace=False)
              )
              (dropout_layer): DropPath()
            )
          )
          (3): PVTEncoderLayer(
            (norm1): LayerNorm((320,), eps=1e-06, elementwise_affine=True)
            (attn): SpatialReductionAttention(
              (attn): MultiheadAttention(
                (out_proj): _LinearWithBias(in_features=320, out_features=320, bias=True)
              )
              (proj_drop): Dropout(p=0.0, inplace=False)
              (dropout_layer): DropPath()
              (sr): Conv2d(320, 320, kernel_size=(2, 2), stride=(2, 2))
              (norm): LayerNorm((320,), eps=1e-06, elementwise_affine=True)
            )
            (norm2): LayerNorm((320,), eps=1e-06, elementwise_affine=True)
            (ffn): MixFFN(
              (layers): Sequential(
                (0): Conv2d(320, 1280, kernel_size=(1, 1), stride=(1, 1))
                (1): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1280)
                (2): GELU()
                (3): Dropout(p=0.0, inplace=False)
                (4): Conv2d(1280, 320, kernel_size=(1, 1), stride=(1, 1))
                (5): Dropout(p=0.0, inplace=False)
              )
              (dropout_layer): DropPath()
            )
          )
          (4): PVTEncoderLayer(
            (norm1): LayerNorm((320,), eps=1e-06, elementwise_affine=True)
            (attn): SpatialReductionAttention(
              (attn): MultiheadAttention(
                (out_proj): _LinearWithBias(in_features=320, out_features=320, bias=True)
              )
              (proj_drop): Dropout(p=0.0, inplace=False)
              (dropout_layer): DropPath()
              (sr): Conv2d(320, 320, kernel_size=(2, 2), stride=(2, 2))
              (norm): LayerNorm((320,), eps=1e-06, elementwise_affine=True)
            )
            (norm2): LayerNorm((320,), eps=1e-06, elementwise_affine=True)
            (ffn): MixFFN(
              (layers): Sequential(
                (0): Conv2d(320, 1280, kernel_size=(1, 1), stride=(1, 1))
                (1): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1280)
                (2): GELU()
                (3): Dropout(p=0.0, inplace=False)
                (4): Conv2d(1280, 320, kernel_size=(1, 1), stride=(1, 1))
                (5): Dropout(p=0.0, inplace=False)
              )
              (dropout_layer): DropPath()
            )
          )
          (5): PVTEncoderLayer(
            (norm1): LayerNorm((320,), eps=1e-06, elementwise_affine=True)
            (attn): SpatialReductionAttention(
              (attn): MultiheadAttention(
                (out_proj): _LinearWithBias(in_features=320, out_features=320, bias=True)
              )
              (proj_drop): Dropout(p=0.0, inplace=False)
              (dropout_layer): DropPath()
              (sr): Conv2d(320, 320, kernel_size=(2, 2), stride=(2, 2))
              (norm): LayerNorm((320,), eps=1e-06, elementwise_affine=True)
            )
            (norm2): LayerNorm((320,), eps=1e-06, elementwise_affine=True)
            (ffn): MixFFN(
              (layers): Sequential(
                (0): Conv2d(320, 1280, kernel_size=(1, 1), stride=(1, 1))
                (1): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1280)
                (2): GELU()
                (3): Dropout(p=0.0, inplace=False)
                (4): Conv2d(1280, 320, kernel_size=(1, 1), stride=(1, 1))
                (5): Dropout(p=0.0, inplace=False)
              )
              (dropout_layer): DropPath()
            )
          )
        )
        (2): LayerNorm((320,), eps=1e-06, elementwise_affine=True)
      )
      (3): ModuleList(
        (0): PatchEmbed(
          (projection): Conv2d(320, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
          (norm): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
        )
        (1): ModuleList(
          (0): PVTEncoderLayer(
            (norm1): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
            (attn): SpatialReductionAttention(
              (attn): MultiheadAttention(
                (out_proj): _LinearWithBias(in_features=512, out_features=512, bias=True)
              )
              (proj_drop): Dropout(p=0.0, inplace=False)
              (dropout_layer): DropPath()
            )
            (norm2): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
            (ffn): MixFFN(
              (layers): Sequential(
                (0): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1))
                (1): Conv2d(2048, 2048, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=2048)
                (2): GELU()
                (3): Dropout(p=0.0, inplace=False)
                (4): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1))
                (5): Dropout(p=0.0, inplace=False)
              )
              (dropout_layer): DropPath()
            )
          )
          (1): PVTEncoderLayer(
            (norm1): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
            (attn): SpatialReductionAttention(
              (attn): MultiheadAttention(
                (out_proj): _LinearWithBias(in_features=512, out_features=512, bias=True)
              )
              (proj_drop): Dropout(p=0.0, inplace=False)
              (dropout_layer): DropPath()
            )
            (norm2): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
            (ffn): MixFFN(
              (layers): Sequential(
                (0): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1))
                (1): Conv2d(2048, 2048, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=2048)
                (2): GELU()
                (3): Dropout(p=0.0, inplace=False)
                (4): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1))
                (5): Dropout(p=0.0, inplace=False)
              )
              (dropout_layer): DropPath()
            )
          )
          (2): PVTEncoderLayer(
            (norm1): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
            (attn): SpatialReductionAttention(
              (attn): MultiheadAttention(
                (out_proj): _LinearWithBias(in_features=512, out_features=512, bias=True)
              )
              (proj_drop): Dropout(p=0.0, inplace=False)
              (dropout_layer): DropPath()
            )
            (norm2): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
            (ffn): MixFFN(
              (layers): Sequential(
                (0): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1))
                (1): Conv2d(2048, 2048, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=2048)
                (2): GELU()
                (3): Dropout(p=0.0, inplace=False)
                (4): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1))
                (5): Dropout(p=0.0, inplace=False)
              )
              (dropout_layer): DropPath()
            )
          )
        )
        (2): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
      )
    )
  )
  init_cfg={'checkpoint': '/home/zhangjp/softlink/Pretrained/pvt_v2_b2.pth'}
  (neck): XPN(
    (lateral_convs): ModuleList(
      (0): ConvModule(
        (conv): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1))
      )
      (1): ConvModule(
        (conv): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1))
      )
      (2): ConvModule(
        (conv): Conv2d(320, 256, kernel_size=(1, 1), stride=(1, 1))
      )
      (3): ConvModule(
        (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
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

########### Model Architecture #################


Process finished with exit code 0
