#!/bin/bash


eta=0.001
gamma=0.1
d=4
for fold in 0;
do
  CUDA_VISIBLE_DEVICES=$((d))  python main_oss.py \
    --benchmark fss \
    --max_sample_iterations 10 \
    --sample-range "(4,6)" \
    --multimask_output 0 \
    --alpha 0.8 --beta 0.2 --exp 1. \
    --num_merging_mask 10 \
    --fold ${fold} --log-root "output/fss/last_f${fold}_${eta}_${gamma}"\
    --device 0 --nested 4 --eta ${eta} --gamma ${gamma}
    
done
#wait