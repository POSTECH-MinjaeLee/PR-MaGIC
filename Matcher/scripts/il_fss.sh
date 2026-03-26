#!/bin/bash
eta=0.0001
gamma=0.1
d=$1
for nshot in 1;
do 
  for fold in 0;
  do
    CUDA_VISIBLE_DEVICES=${d} python main_oss_IL.py \
      --benchmark dis\
      --nshot ${nshot} \
      --max_sample_iterations 30 \
      --sample-range "(4,6)" \
      --multimask_output 0 \
      --alpha 0.8 --beta 0.2 --exp 1. \
      --num_merging_mask 10 \
      --num_centers 5\
      --fold ${fold} --log-root "CVPR/dis/orig-${eta}_${gamma}_${nshot}"\
      --device 0 --nested 6 --eta ${eta} --gamma ${gamma} \
      --alpha_list "0.0,0.1,0.25,0.5" --beta_list "0.0,0.05,0.1,0.2"
  done
done

eta=0.001
gamma=0.1
for nshot in 1 5;
do 
  for fold in 0;
  do
    CUDA_VISIBLE_DEVICES=${d} python main_oss_IL.py \
      --benchmark fss \
      --nshot ${nshot} \
      --max_sample_iterations 30 \
      --sample-range "(4,6)" \
      --multimask_output 0 \
      --alpha 0.8 --beta 0.2 --exp 1. \
      --num_merging_mask 10 \
      --fold ${fold} --log-root "CVPR/fss/${eta}_${gamma}_${nshot}"\
      --device 0 --nested 6 --eta ${eta} --gamma ${gamma} \
      --alpha_list "0.0,0.1,0.25,0.5" --beta_list "0.0,0.05,0.1,0.2"
  done
done

eta=0.0001
gamma=0.1
for nshot in 1 5;
do
    for fold in 0 1 2 3;
    do
    CUDA_VISIBLE_DEVICES=${d} python main_oss_IL.py  \
        --benchmark paco_part \
        --max_sample_iterations 30 \
        --sample-range "(4,6)" \
        --nshot ${nshot} \
        --alpha 0.5 --beta 0.5 --exp 0. \
        --num_merging_mask 5 \
        --box_nms_thresh 0.65 \
        --use_score_filter \
        --num_centers 5 \
        --eta ${eta} --gamma ${gamma} --fold ${fold} --log-root "CVPR/paco/fold:${fold}_${eta}_${gamma}_${nshot}" \
        --nested 6 \
        --alpha_list "0.0,0.1,0.25,0.5" --beta_list "0.0,0.05,0.1,0.2"
    done
done