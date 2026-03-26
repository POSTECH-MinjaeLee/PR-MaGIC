#!/bin/bash
eta=0.001
gamma=0.1 
device=$1
for nshot in 1 5;
do
    for fold in 0 1 2 3;
    do
        CUDA_VISIBLE_DEVICES=${device} python main_oss_IL.py  \
        --benchmark coco \
        --nshot ${nshot}\
        --max_sample_iterations 30 \
        --sample-range "(4,6)" \
        --multimask_output 0 \
        --alpha 1.0 --beta 0. --exp 0. \
        --num_merging_mask 9  \
        --use_score_filter \
        --box_nms_thresh 0.65 \
        --eta ${eta} --gamma ${gamma} --fold ${fold} --log-root "CVPR/coco/device_${device}_f${fold}_${eta}_${gamma}_${nshot}" --nested 6 \
        --alpha_list "0.0,0.1,0.25,0.5" --beta_list "0.0,0.05,0.1,0.2"
    done 
done

#!/bin/bash
eta=0.0001
gamma=0.1
for nshot in 1 5;
do
  for fold in 0 1 2 3;
  do
    CUDA_VISIBLE_DEVICES=${device} python main_oss_IL.py \
      --benchmark pascal_part \
      --max_sample_iterations 30 \
      --sample-range "(4,6)" \
      --box_nms_thresh 0.65 \
      --nshot ${nshot} \
      --use_score_filter \
      --alpha 0.5 --beta 0.5 --exp 0. \
      --num_merging_mask 5 \
      --num_centers 5 \
      --fold ${fold} --log-root "CVPR/pascal/fold:${fold}_${eta}_${gamma}_${nshot}"\
      --device 0 --nested 6 --eta ${eta} --gamma ${gamma} \
      --alpha_list "0.0,0.1,0.25,0.5" --beta_list "0.0,0.05,0.1,0.2"
  done
done