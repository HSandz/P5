# Run with $ bash scripts/pretrain_P5_base_movielens.sh 4
# For single GPU, run with $ bash scripts/pretrain_P5_base_movielens.sh 1

#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

name=movielens-base

output=snap/$name

PYTHONPATH=$PYTHONPATH:./src \
python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_port 12347 \
    src/pretrain.py \
        --distributed --multiGPU \
        --seed 2022 \
        --train ml100k \
        --valid ml100k \
        --batch_size 16 \
        --optim adamw \
        --warmup_ratio 0.05 \
        --lr 1e-3 \
        --num_workers 4 \
        --clip_grad_norm 1.0 \
        --losses 'rating,sequential,traditional' \
        --backbone 't5-base' \
        --output $output ${@:2} \
        --epoch 20 \
        --max_text_length 512 \
        --gen_max_length 64 \
        --whole_word_embed > $name.log
