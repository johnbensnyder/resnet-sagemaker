#!/usr/bin/env bash

GPU_COUNT=`nvidia-smi --query-gpu=name --format=csv,noheader | wc -l`

source activate tensorflow2_latest_p37

CONDA_PYTHON=`which python`

mpirun --allow-run-as-root --tag-output --mca plm_rsh_no_tree_spawn 1 \
    --mca btl_tcp_if_exclude lo,docker0 \
    -np $GPU_COUNT -H localhost:$GPU_COUNT \
    -x NCCL_DEBUG=VERSION \
    -x LD_LIBRARY_PATH \
    -x PATH \
    --oversubscribe \
    $CONDA_PYTHON train.py \
    --train_data_dir ~/data/imagenet/tfrecord/train \
    --validation_data_dir ~/data/imagenet/tfrecord/validation \
    --batch_size 2048 \
    --num_epochs 120 \
    --model_dir ~/model \
    --learning_rate 0.1 \
    --momentum 0.9 \
    --label_smoothing 0.1 \
    --mixup_alpha 0.2 \
    --l2_weight_decay 1.0e-5 \
    --fp16 True \
    --xla True \
    --tf32 True \
    --model resnet152v1_d
