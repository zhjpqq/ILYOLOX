#!/usr/bin/env bash

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
#export CUDA_VISIBLE_DEVICES=0,1,2,3
#export CUDA_VISIBLE_DEVICES=4,5,6,7
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

startTime=`date +"%Y-%m-%d %H:%M:%S"`

# nohup tools/dist_train_increment.sh &
# nohup tools/dist_train_increment.sh 1>$expdir/yoloy_r50_stg8b16_qoqo_il44r_d5_mxone/nohup51 2>&1 &
# nohup tools/dist_train_increment.sh 1>$expdir/yoloy_r50_stg8b16_qoqo_il2222_dx_msone/nohup51 2>&1 &
# nohup tools/dist_train_increment.sh 1>$expdir/yoloy_r50_stg8b16_qoqo_il71_u1_hard/nohup55 2>&1 &
# nohup tools/dist_train_increment.sh 1>$expdir/yoloy_r50_st4gb16_qoqo_il53_x1_hard/nohup11 2>&1 &
# nohup tools/dist_train_increment.sh 1>$expdir/yoloy_r50_aug1g8b16_qoqo_il80_u1/nohup51 2>&1 &
# nohup tools/dist_train_increment.sh 1>$expdir/common_exp_il/nohup 2>&1 &

#CONFIG=${1:-'/home/softlink/zhjproj/now-projects/xmdet220/configs/yolof/yolof_resnet_qoqo_il.py'}
#WORKDIR=${2:-'/home/softlink/zhjpexp/yolof-r18-stst-qoqo-il80-t1'}
CONFIG=${1:-'/home/softlink/zhjproj/now-projects/xmdet220/configs/yoloy/yoloy_resnet_qoqo_il.py'}
#WORKDIR=${2:-'/home/softlink/zhjpexp/yoloy_r50_st4gb16_qoqo_il53_x1_hard'}
#WORKDIR=${2:-'/home/softlink/zhjpexp/yoloy_r50_stg8b16_qoqo_il41111_dx_mscto'}
WORKDIR=${2:-'/home/softlink/zhjpexp/yoloy_r50_stg8b16_qoqo_il71_u1_hard'}
#CONFIG=${1:-'/home/softlink/zhjproj/now-projects/xmdet220/configs/yoloy/yoloy_resnet_wrxt_il.py'}
#WORKDIR=${2:-'/home/softlink/zhjpexp/yoloy_r18_stst_wrxt_il54_h1'}
#CONFIG=${1:-'/home/softlink/zhjproj/now-projects/xmdet220/configs/aaamixer/aaamixer_resnet_qoqo_il20.py'}
#WORKDIR=${2:-'/home/softlink/zhjpexp/amixer_r18_stqo_df2521v2sfm10stg3_il20_up1_d5g4b12'}
#CONFIG=${1:-'/home/softlink/zhjproj/now-projects/xmdet220/configs/aaamixer/aaamixer_resnet_qoqo_il.py'}
#WORKDIR=${2:-'/home/softlink/zhjpexp/amixer_r18_stqo_df2521v2sfm10stg3_il20it2_v6_d5g4b12_scsf2lsf1hxpnpl15gi2'}

#WORKDIR=${2:-'/home/softlink/zhjpexp/common_exp_il'}
CHECKPOINT=${4:-''}

GPUS=${3:-8}
PORT=${PORT:-36198}

if [ ! -d "$WORKDIR" ];then
  mkdir $WORKDIR
  touch $WORKDIR'/nohup'
  echo "??????????????????nohup??????: $WORKDIR"
else
  touch $WORKDIR'/nohup'
  echo "????????????nohup?????????: $WORKDIR"
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
echo "???????????????: $sumHours ?????????$sumMinutes ??????."
