<div align="center">
  <img src="resources/mmdet-logo.png" width="600"/>
  <div>&nbsp;</div>
  <div align="center">
    <b>OpenMMLab website</b>
    <sup>
      <a href="https://openmmlab.com">
        <i>HOT</i>
      </a>
    </sup>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <b>OpenMMLab platform</b>
    <sup>
      <a href="https://platform.openmmlab.com">
        <i>TRY IT OUT</i>
      </a>
    </sup>
  </div>
  <div>&nbsp;</div>

[![PyPI](https://img.shields.io/pypi/v/mmdet)](https://pypi.org/project/mmdet)
[![docs](https://img.shields.io/badge/docs-latest-blue)](https://mmdetection.readthedocs.io/en/latest/)
[![codecov](https://codecov.io/gh/open-mmlab/mmdetection/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmdetection)
[![license](https://img.shields.io/github/license/open-mmlab/mmdetection.svg)](https://github.com/open-mmlab/mmdetection/blob/master/LICENSE)
[![open issues](https://isitmaintained.com/badge/open/open-mmlab/mmdetection.svg)](https://github.com/open-mmlab/mmdetection/issues)

  <img src="https://user-images.githubusercontent.com/12907710/137271636-56ba1cd2-b110-4812-8221-b4c120320aa9.png"/>

[📘Documentation](https://mmdetection.readthedocs.io/en/v2.21.0/) |
[🛠️Installation](https://mmdetection.readthedocs.io/en/v2.21.0/get_started.html) |
[👀Model Zoo](https://mmdetection.readthedocs.io/en/v2.21.0/model_zoo.html) |
[🆕Update News](https://mmdetection.readthedocs.io/en/v2.21.0/changelog.html) |
[🚀Ongoing Projects](https://github.com/open-mmlab/mmdetection/projects) |
[🤔Reporting Issues](https://github.com/open-mmlab/mmdetection/issues/new/choose)

</div>

## Incremental(Continual) Object Detection Frame based on MMDetection

MMDetection is an open source object detection toolbox based on PyTorch. It is
a part of the [OpenMMLab](https://openmmlab.com/) project.

The master branch works with **PyTorch 1.5+**.

## Incremental YOLOX

<img src="ilyolox/ilyolox-overall-architecture.png">

<img src="ilyolox/coco-mlti-step-il.png">

<img src="ilyolox/coco-2xsteps.png">

<img src="ilyolox/cocox4-il.png">

## Traing & Evaluation

```
nohup tools/dist_train_increment.sh 1>$expdir/common_exp_il/nohup 2>&1 &
```

```
tools/xdist_test.sh
```

## Checkpoint Dowload

Google Drive Preparing!

## License

This project is released under the [Apache 2.0 license](LICENSE).
