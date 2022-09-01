
============== model arch start ==============
DETR(
  (backbone): ResNet(
    (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu): ReLU(inplace=True)
    (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    (layer4): ResLayer(
      (0): BasicBlock(
        (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (downsample): Sequential(
          (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
          (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): BasicBlock(
        (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
    )
  )
  init_cfg={'type': 'Pretrained', 'checkpoint': '/home/softlink/Pretrained/resnet18-5c106cde.pth'}
  (bbox_head): DETRHead(
    (loss_cls): CrossEntropyLoss()
    (loss_bbox): L1Loss()
    (loss_iou): GIoULoss()
    (activate): ReLU(inplace=True)
    (positional_encoding): SinePositionalEncoding(num_feats=128, temperature=10000, normalize=True, scale=6.283185307179586, eps=1e-06)
    (transformer): Transformer(
      (encoder): DetrTransformerEncoder(
        (layers): ModuleList(
          (0): BaseTransformerLayer(
            (attentions): ModuleList(
              (0): MultiheadAttention(
                (attn): MultiheadAttention(
                  (out_proj): _LinearWithBias(in_features=256, out_features=256, bias=True)
                )
                (proj_drop): Dropout(p=0.0, inplace=False)
                (dropout_layer): Dropout(p=0.1, inplace=False)
              )
            )
            (ffns): ModuleList(
              (0): FFN(
                (activate): ReLU(inplace=True)
                (layers): Sequential(
                  (0): Sequential(
                    (0): Linear(in_features=256, out_features=2048, bias=True)
                    (1): ReLU(inplace=True)
                    (2): Dropout(p=0.1, inplace=False)
                  )
                  (1): Linear(in_features=2048, out_features=256, bias=True)
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
              (0): MultiheadAttention(
                (attn): MultiheadAttention(
                  (out_proj): _LinearWithBias(in_features=256, out_features=256, bias=True)
                )
                (proj_drop): Dropout(p=0.0, inplace=False)
                (dropout_layer): Dropout(p=0.1, inplace=False)
              )
            )
            (ffns): ModuleList(
              (0): FFN(
                (activate): ReLU(inplace=True)
                (layers): Sequential(
                  (0): Sequential(
                    (0): Linear(in_features=256, out_features=2048, bias=True)
                    (1): ReLU(inplace=True)
                    (2): Dropout(p=0.1, inplace=False)
                  )
                  (1): Linear(in_features=2048, out_features=256, bias=True)
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
              (0): MultiheadAttention(
                (attn): MultiheadAttention(
                  (out_proj): _LinearWithBias(in_features=256, out_features=256, bias=True)
                )
                (proj_drop): Dropout(p=0.0, inplace=False)
                (dropout_layer): Dropout(p=0.1, inplace=False)
              )
            )
            (ffns): ModuleList(
              (0): FFN(
                (activate): ReLU(inplace=True)
                (layers): Sequential(
                  (0): Sequential(
                    (0): Linear(in_features=256, out_features=2048, bias=True)
                    (1): ReLU(inplace=True)
                    (2): Dropout(p=0.1, inplace=False)
                  )
                  (1): Linear(in_features=2048, out_features=256, bias=True)
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
              (0): MultiheadAttention(
                (attn): MultiheadAttention(
                  (out_proj): _LinearWithBias(in_features=256, out_features=256, bias=True)
                )
                (proj_drop): Dropout(p=0.0, inplace=False)
                (dropout_layer): Dropout(p=0.1, inplace=False)
              )
            )
            (ffns): ModuleList(
              (0): FFN(
                (activate): ReLU(inplace=True)
                (layers): Sequential(
                  (0): Sequential(
                    (0): Linear(in_features=256, out_features=2048, bias=True)
                    (1): ReLU(inplace=True)
                    (2): Dropout(p=0.1, inplace=False)
                  )
                  (1): Linear(in_features=2048, out_features=256, bias=True)
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
              (0): MultiheadAttention(
                (attn): MultiheadAttention(
                  (out_proj): _LinearWithBias(in_features=256, out_features=256, bias=True)
                )
                (proj_drop): Dropout(p=0.0, inplace=False)
                (dropout_layer): Dropout(p=0.1, inplace=False)
              )
            )
            (ffns): ModuleList(
              (0): FFN(
                (activate): ReLU(inplace=True)
                (layers): Sequential(
                  (0): Sequential(
                    (0): Linear(in_features=256, out_features=2048, bias=True)
                    (1): ReLU(inplace=True)
                    (2): Dropout(p=0.1, inplace=False)
                  )
                  (1): Linear(in_features=2048, out_features=256, bias=True)
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
              (0): MultiheadAttention(
                (attn): MultiheadAttention(
                  (out_proj): _LinearWithBias(in_features=256, out_features=256, bias=True)
                )
                (proj_drop): Dropout(p=0.0, inplace=False)
                (dropout_layer): Dropout(p=0.1, inplace=False)
              )
            )
            (ffns): ModuleList(
              (0): FFN(
                (activate): ReLU(inplace=True)
                (layers): Sequential(
                  (0): Sequential(
                    (0): Linear(in_features=256, out_features=2048, bias=True)
                    (1): ReLU(inplace=True)
                    (2): Dropout(p=0.1, inplace=False)
                  )
                  (1): Linear(in_features=2048, out_features=256, bias=True)
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
      (decoder): DetrTransformerDecoder(
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
              (1): MultiheadAttention(
                (attn): MultiheadAttention(
                  (out_proj): _LinearWithBias(in_features=256, out_features=256, bias=True)
                )
                (proj_drop): Dropout(p=0.0, inplace=False)
                (dropout_layer): Dropout(p=0.1, inplace=False)
              )
            )
            (ffns): ModuleList(
              (0): FFN(
                (activate): ReLU(inplace=True)
                (layers): Sequential(
                  (0): Sequential(
                    (0): Linear(in_features=256, out_features=2048, bias=True)
                    (1): ReLU(inplace=True)
                    (2): Dropout(p=0.1, inplace=False)
                  )
                  (1): Linear(in_features=2048, out_features=256, bias=True)
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
              (1): MultiheadAttention(
                (attn): MultiheadAttention(
                  (out_proj): _LinearWithBias(in_features=256, out_features=256, bias=True)
                )
                (proj_drop): Dropout(p=0.0, inplace=False)
                (dropout_layer): Dropout(p=0.1, inplace=False)
              )
            )
            (ffns): ModuleList(
              (0): FFN(
                (activate): ReLU(inplace=True)
                (layers): Sequential(
                  (0): Sequential(
                    (0): Linear(in_features=256, out_features=2048, bias=True)
                    (1): ReLU(inplace=True)
                    (2): Dropout(p=0.1, inplace=False)
                  )
                  (1): Linear(in_features=2048, out_features=256, bias=True)
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
              (1): MultiheadAttention(
                (attn): MultiheadAttention(
                  (out_proj): _LinearWithBias(in_features=256, out_features=256, bias=True)
                )
                (proj_drop): Dropout(p=0.0, inplace=False)
                (dropout_layer): Dropout(p=0.1, inplace=False)
              )
            )
            (ffns): ModuleList(
              (0): FFN(
                (activate): ReLU(inplace=True)
                (layers): Sequential(
                  (0): Sequential(
                    (0): Linear(in_features=256, out_features=2048, bias=True)
                    (1): ReLU(inplace=True)
                    (2): Dropout(p=0.1, inplace=False)
                  )
                  (1): Linear(in_features=2048, out_features=256, bias=True)
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
              (1): MultiheadAttention(
                (attn): MultiheadAttention(
                  (out_proj): _LinearWithBias(in_features=256, out_features=256, bias=True)
                )
                (proj_drop): Dropout(p=0.0, inplace=False)
                (dropout_layer): Dropout(p=0.1, inplace=False)
              )
            )
            (ffns): ModuleList(
              (0): FFN(
                (activate): ReLU(inplace=True)
                (layers): Sequential(
                  (0): Sequential(
                    (0): Linear(in_features=256, out_features=2048, bias=True)
                    (1): ReLU(inplace=True)
                    (2): Dropout(p=0.1, inplace=False)
                  )
                  (1): Linear(in_features=2048, out_features=256, bias=True)
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
              (1): MultiheadAttention(
                (attn): MultiheadAttention(
                  (out_proj): _LinearWithBias(in_features=256, out_features=256, bias=True)
                )
                (proj_drop): Dropout(p=0.0, inplace=False)
                (dropout_layer): Dropout(p=0.1, inplace=False)
              )
            )
            (ffns): ModuleList(
              (0): FFN(
                (activate): ReLU(inplace=True)
                (layers): Sequential(
                  (0): Sequential(
                    (0): Linear(in_features=256, out_features=2048, bias=True)
                    (1): ReLU(inplace=True)
                    (2): Dropout(p=0.1, inplace=False)
                  )
                  (1): Linear(in_features=2048, out_features=256, bias=True)
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
              (1): MultiheadAttention(
                (attn): MultiheadAttention(
                  (out_proj): _LinearWithBias(in_features=256, out_features=256, bias=True)
                )
                (proj_drop): Dropout(p=0.0, inplace=False)
                (dropout_layer): Dropout(p=0.1, inplace=False)
              )
            )
            (ffns): ModuleList(
              (0): FFN(
                (activate): ReLU(inplace=True)
                (layers): Sequential(
                  (0): Sequential(
                    (0): Linear(in_features=256, out_features=2048, bias=True)
                    (1): ReLU(inplace=True)
                    (2): Dropout(p=0.1, inplace=False)
                  )
                  (1): Linear(in_features=2048, out_features=256, bias=True)
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
        (post_norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
      )
    )
    (input_proj): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
    (fc_cls): Linear(in_features=256, out_features=11, bias=True)
    (reg_ffn): FFN(
      (activate): ReLU(inplace=True)
      (layers): Sequential(
        (0): Sequential(
          (0): Linear(in_features=256, out_features=256, bias=True)
          (1): ReLU(inplace=True)
          (2): Dropout(p=0.0, inplace=False)
        )
        (1): Linear(in_features=256, out_features=256, bias=True)
        (2): Dropout(p=0.0, inplace=False)
      )
      (dropout_layer): Identity()
    )
    (fc_reg): Linear(in_features=256, out_features=4, bias=True)
    (query_embedding): Embedding(100, 256)
  )
)
============== model arch end ==============
