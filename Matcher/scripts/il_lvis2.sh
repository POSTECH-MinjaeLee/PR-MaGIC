#!/bin/bash
eta=0.001
gamma=0.1 
device=$1
for nshot in 1 5;
do
  for fold in 5 6 7 8 9;
  do
    CUDA_VISIBLE_DEVICES=${device} python main_oss_IL.py  \
      --benchmark lvis \
      --nshot ${nshot} \
      --max_sample_iterations 30 \
      --sample-range "(4,6)" \
      --multimask_output 0 \
      --alpha 1.0 --beta 0. --exp 0. \
      --num_merging_mask 9 \
      --use_score_filter \
      --box_nms_thresh 0.65 \
      --eta ${eta} --gamma ${gamma} --fold ${fold} --log-root "CVPR/lvis/fold:${fold}_${eta}_${gamma}_${nshot}" --nested 6 \
      --alpha_list "0.0,0.1,0.25,0.5" --beta_list "0.0,0.05,0.1,0.2"
  done
done
#wait