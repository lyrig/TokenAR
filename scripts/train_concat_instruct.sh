#!/bin/bash
# 8.24 Instruct Token Experiment Bash Code
# Aim:
# 1. Use Single learnable Instruct Token Embedding in all situations, and test whether the performance would be better with more instruct token channel.
# 2. Use Multiple learnable Instruct Token Embedding in many situations, and test whether the token can instruct the model's output
# 3. Use Multiple Learnable Instruct Token Embedding to make model generate different types of images.
NUM_GPU=8
BATCHSIZE=4
INSTRUCT_TOKEN_NUM=120
# Now Step 1.
torchrun --nnodes=1 --nproc-per-node=$NUM_GPU --master-port=25001 \
        autoregressive/train/train_edit_multicond_plus.py \
        --output-dir "checkpoints/train/Concat_Instruct1_${INSTRUCT_TOKEN_NUM}_0.5_subject10K" \
        --vq-ckpt your_path/ckpt/llamagen_t2i/vq_ds16_t2i.pt \
        --t5-path your_path/ckpt \
        --image-size 512 \
        --global-batch-size $((NUM_GPU * BATCHSIZE)) \
        --gpt-model GPT-XL \
        --gpt-mode 'joint_cls_emb' \
        --gpt-ckpt your_path/ckpt/llamagen_t2i/t2i_XL_stage2_512.pt \
        --num-workers 4 \
        --epochs 50 \
        --use-distill \
        --distill-mode dinov2\
        --distill-loss-weight 0.5 2.0\
        --dataset-list multicond++ \
        --unicombine-path your_path/data/unzipSubject200K \
        --unicombine-prob 0.0 \
        --use_wandb \
        --ref_index_embed \
        --max_ref_num 4 \
        --concat-target \
        --instruct-token-mode casual \
        --instruct-token-num "${INSTRUCT_TOKEN_NUM}" \
