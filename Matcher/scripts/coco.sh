#!/bin/bash
eta=0.0001
gamma=0.01 
device=1
for fold in 0 1 2 3;
do
    CUDA_VISIBLE_DEVICES=${device} python main_oss.py  \
    --benchmark coco \
    --max_sample_iterations 30 \
    --sample-range "(1,6)" \
    --multimask_output 0 \
    --alpha 1.0 --beta 0. --exp 0. \
    --num_merging_mask 9  \
    --eta ${eta} --gamma ${gamma} --fold ${fold} --log-root "output/coco/device_${device}_f${fold}_${eta}_${gamma}" --nested 5
done
#wait 
#--use_dense_mask 1
