#!/usr/bin/env bash

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
#export CUDA_VISIBLE_DEVICES=4,5,6,7

startTime=`date +"%Y-%m-%d %H:%M:%S"`

# nohup tools/xxx_train_increment.sh 1>$expdir/common_exp_il/nohup 2>&1 &

#CONFIG=${1:-'/home/zhangjp/projects/now-projects/xmdet220/configs/yolof/yolof_resnet_qoqo_il.py'}
#WORKDIR=${2:-'/home/softlink/zhjpexp/yolof-r18-stst-qoqo-il80-t1'}
CONFIG=${1:-'/home/zhangjp/projects/now-projects/xmdet220/configs/yoloy/yoloy_resnet_qoqo_ilxxx.py'}
#WORKDIR=${2:-'/home/softlink/zhjpexp/common_exp_il'}
#CONFIG=${1:-'/home/zhangjp/projects/now-projects/xmdet220/configs/yoloy/yoloy_resnet_wrxt_il.py'}
#WORKDIR=${2:-'/home/softlink/zhjpexp/yoloy_r18_stst_wrxt_il54_hpnpcsce1box0iou5alp07mssm02'}
#CONFIG=${1:-'/home/zhangjp/projects/now-projects/xmdet220/configs/aaamixer/aaamixer_resnet_qoqo_il20.py'}
#WORKDIR=${2:-'/home/softlink/zhjpexp/amixer_r18_stqo_df2521v2sfm10stg3_il20_up1_d5g4b12'}
#CONFIG=${1:-'/home/zhangjp/projects/now-projects/xmdet220/configs/aaamixer/aaamixer_resnet_qoqo_il.py'}
#WORKDIR=${2:-'/home/softlink/zhjpexp/amixer_r18_stqo_df2521v2sfm10stg3_il20it2_v6_d5g4b12_scsf2lsf1hxpnpl15gi2'}

WORKDIR=${2:-'/home/softlink/zhjpexp/common_exp_il'}
CHECKPOINT=${4:-''}

GPUS=${3:-4}
PORT=${PORT:-36312}

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
