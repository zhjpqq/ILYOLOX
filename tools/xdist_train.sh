#!/usr/bin/env bash

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
# export CUDA_VISIBLE_DEVICES=0, 1
startTime=`date +"%Y-%m-%d %H:%M:%S"`

# nohup tools/xdist_train.sh 1>$expdir/aaamixer_r50_qoqo_stst-v3-wl1.5-wdf20-softmax/nohup2 2>&1 &
# nohup tools/xdist_train.sh 1>$expdir/common_exp/nohup 2>&1 &

#CONFIG=${1:-'/home/zhangjp/projects/now-projects/xmdet220/configs/yoloy/yoloy_resnet_qoqo.py'}
#WORKDIR=${2:-'/home/softlink/zhjpexp/yoloy-r18-stst-qoqo'}
#CHECKPOINT=${4:-'/home/softlink/zhjpexp/yoloy-r18-stst-qoqo/latest.pth'}

#CONFIG=${1:-'/home/zhangjp/projects/now-projects/xmdet220/configs/adamixer/adamixer_resnet_qoqo.py'}
#WORKDIR=${2:-'/home/softlink/zhjpexp/adamixer_r50_qoqo_stst-v1'}

CONFIG=${1:-'/home/zhangjp/projects/now-projects/xmdet220/configs/aaamixer/aaamixer_resnet_qoqo.py'}
#WORKDIR=${2:-'/home/softlink/zhjpexp/aaamixer_r50_qoqo_stst-v3-wl1.5-wdf20-softmax'}
#CHECKPOINT=${4:-'/home/softlink/zhjpexp/aaamixer_r50_qoqo_stst-v3-wl1.5-wdf20-softmax/epoch_4.pth'}

WORKDIR=${2:-'/home/softlink/zhjpexp/common_exp'}
CHECKPOINT=${4:-''}
GPUS=${3:-4}
PORT=${PORT:-29500}

if [ ! -d "$WORKDIR" ];then
  mkdir $WORKDIR
  touch $WORKDIR'/nohup'
  echo "创建文件夹和nohup成功: $WORKDIR"
else
  touch $WORKDIR'/nohup'
  echo "文件夹和nohup已存在: $WORKDIR"
fi

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
          $(dirname "$0")/xxtrain.py \
          --config=$CONFIG \
          --resume-from=$CHECKPOINT \
          --work-dir=$WORKDIR \
          --launcher=pytorch ${@:3}

endTime=`date +"%Y-%m-%d %H:%M:%S"`
st=`date -d  "$startTime" +%s`
et=`date -d  "$endTime" +%s`

sumHours=$((($et-$st)/3600))
sumMinutes=$((($et-$st)%60))
echo "运行总时间: $sumHours 小时，$sumMinutes 分钟."

