## Incremental (Continual) Object Detection Frame based on MMDetection

# Center Resampling and Collaborative Knowledge Distillation Strategies for Class Incremental Object Detection

[comment]: <> (MMDetection is an open source object detection toolbox based on PyTorch. It is)

[comment]: <> (a part of the [OpenMMLab]&#40;https://openmmlab.com/&#41; project.)

[comment]: <> (The master branch works with **PyTorch 1.5+**.)

[![PyPI](https://img.shields.io/pypi/v/mmdet)](https://pypi.org/project/mmdet)
[![docs](https://img.shields.io/badge/docs-latest-blue)](https://mmdetection.readthedocs.io/en/latest/)
[![codecov](https://codecov.io/gh/open-mmlab/mmdetection/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmdetection)
[![license](https://img.shields.io/github/license/open-mmlab/mmdetection.svg)](https://github.com/open-mmlab/mmdetection/blob/master/LICENSE)
[![open issues](https://isitmaintained.com/badge/open/open-mmlab/mmdetection.svg)](https://github.com/open-mmlab/mmdetection/issues)


## 1. Abstract

[comment]: <> (Catastrophic forgetting is the key problem in Class Incremental Object Detection&#40;CIOD&#41; task. )

[comment]: <> (Knowledge distillation has been proved to be an effective way to solve this problem.However, )

[comment]: <> (most previous works need to combine several distillation methods including feature, )

[comment]: <> (classification, location and relation to work together. In this paper, we only use classification )

[comment]: <> (distillation to build incremental detector. First, an adaptive structured knowledge selection strategy )

[comment]: <> (is proposed to make a better trade-off between the quality and quantity of teacher outputs, thus enhancing )

[comment]: <> (the excavation of detection knowledge from teacher model. Second, a collaborative knowledge )

[comment]: <> (transfer strategy is proposed to accelerate the collaboratively transfer between classification and location from teacher model to student )

[comment]: <> (model. We demonstrate that the reasonable knowledge selection and transfer strategy are the keys to overcome )

[comment]: <> (catastrophic forgetting for CIOD task. Extensive experiments conducted on COCO2017 demonstrate )

[comment]: <> (that our method achieves state-of-the-art results under various scenarios, which gives remarkable )

[comment]: <> (improvements by large margins than the previous best results. Code is available at https://github.com/zhjpqq/ILYOLOX.)

Catastrophic forgetting is a key problem in Class Incremental object detection (CIOD). Knowledge distillation 
has been proved to be an effective way to solve this problem. **However, most previous works combine several 
distillation methods including feature, classification, location and relation into a complex framework to work together to 
relieve catastrophic forgetting. In this paper, 
we only use classification distillation to build incremental detectors**. Firstly, an adaptive structured knowledge 
selection strategy is proposed to better balance the quality and quantity of teacher outputs, thus enhancing the 
mining of detection knowledge from teacher model. Secondly, a collaborative knowledge transfer strategy is 
proposed to accelerate the collaborative transfer of classification and location knowledge to student model. 
Our work shows that reasonable knowledge selection and transfer strategies are the key to overcome catastrophic 
forgetting and improve the performance of CIOD task. Extensive experiments on COCO2017 show that our method achieves 
the state-of-art results under various incremental learning scenarios and significantly narrows the performance gap 
between incremental learning and overall learning. The code will be avaliable at https://github.com/zhjpqq/ILYOLOX.

## 2. Network Architecture

![ilyolox/ilyolox-overall-architecture.png](https://img-blog.csdnimg.cn/67a88f7bf1be4dbca3fe6812bed37d66.png)

![ilyolox/ilyolox-overall-architecture.png](ilyolox/ilyolox-overall-architecture.png)


## 3. Overall Performance

[comment]: <> (![ilyolox/coco-mlti-step-il.png]&#40;ilyolox/coco-mlti-step-il.png&#41;)

![ilyolox/coco-mlti-step-il.png](https://img-blog.csdnimg.cn/7bfdfd09c9904baeb500da93eb8ff12b.png)


###**ERD** is proposed in CVPR 2022: Overcoming Catastrophic Forgetting in Incremental Object Detection via Elastic Response Distillation.


[comment]: <> (![ilyolox/coco-2xsteps.png]&#40;ilyolox/coco-2xsteps.png&#41;)

![ilyolox/coco-2xsteps.png](https://img-blog.csdnimg.cn/49e0ffa12cb245f19688ebce53cf5070.png)


## 4. Traing & Evaluation

```
nohup tools/dist_train_increment.sh 1>$expdir/common_exp_il/nohup 2>&1 &
```

```
tools/xdist_test.sh
```

## 5. Checkpoint Dowload

Google Drive Preparing!  


## 6. License

This project is released under the [Apache 2.0 license](LICENSE).
