#!/usr/bin/env bash

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
#export CUDA_VISIBLE_DEVICES=4,5,6,7
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

startTime=`date +"%Y-%m-%d %H:%M:%S"`

# nohup tools/dist_train_increment_wrxt.sh &
# nohup tools/dist_train_increment_wrxt.sh 1>$expdir/yoloy_r18_stst_wrxt_g4b8_il54_u1_ohot/nohup036511v5 2>&1 &
# nohup tools/dist_train_increment_wrxt.sh 1>$expdir/yoloy_r18_stst_wrxt_g4b8_il54_u1_hybr/nohup036511v7 2>&1 &
# nohup tools/dist_train_increment_wrxt.sh 1>$expdir/common_exp_il/nohup 2>&1 &

CONFIG=${1:-'/home/softlink/zhjproj/now-projects/xmdet220/configs/yoloy/yoloy_resnet_wrxt_il.py'}
WORKDIR=${2:-'/home/softlink/zhjpexp/yoloy_r18_stst_wrxt_g4b8_il54_u1_hybr'}
#CONFIG=${1:-'/home/softlink/zhjproj/now-projects/xmdet220/configs/yoloy/yoloy_resnet_wrxt_il.py'}

#WORKDIR=${2:-'/home/softlink/zhjpexp/common_exp_il'}
CHECKPOINT=${4:-''}

GPUS=${3:-4}
PORT=${PORT:-36198}

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
          $(dirname "$0")/train_increment.py \
          --config=$CONFIG \
          --work-dir=$WORKDIR \
          --resume-from=$CHECKPOINT \
          --launcher=pytorch ${@:3}

endTime=`date +"%Y-%m-%d %H:%M:%S"`
st=`date -d  "$startTime" +%s`
et=`date -d  "$endTime" +%s`

sumHours=$((($et-$st)/3600))
sumMinutes=$((($et-$st)%60))
echo "运行总时间: $sumHours 小时，$sumMinutes 分钟."
