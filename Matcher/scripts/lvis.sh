#!/bin/bash
eta=0.0001
gamma=0.01 
device=3
for fold in 0 1 2 3 4 5 6 7 8 9;
do
  CUDA_VISIBLE_DEVICES=${device} python main_oss.py  \
    --benchmark lvis \
    --max_sample_iterations 30 \
    --sample-range "(1,6)" \
    --multimask_output 0 \
    --alpha 1.0 --beta 0. --exp 0. \
    --num_merging_mask 9 \
    --eta ${eta} --gamma ${gamma} --fold ${fold} --log-root "output/lvis/device${device}_f${fold}_${eta}_${gamma}" --nested 5
done
#wait