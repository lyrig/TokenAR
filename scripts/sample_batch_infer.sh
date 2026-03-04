#!/bin/bash

# Batch inference script for TokenAR multi-condition generation.
# Modify the paths below before running.

set -e

DEVICE="cuda"
BATCH_SIZE=8
NUM_WORKERS=4
CFG_SCALE=3
MAX_REF_NUM=4
INSTRUCT_TOKEN_NUM=120

DATASET_PATH="your_path/data/unzipSubject200K_eval"
GPT_CKPT="your_path/checkpoints/train/model.pt" # 训练得到的ckpt
VQ_CKPT="your_path/ckpt/llamagen_t2i/vq_ds16_t2i.pt"
T5_PATH="your_path/ckpt"
LOG_DIR="your_path/checkpoints/eval"
ADDITIONAL_INFO="batch_eval"

python autoregressive/sample/sample_edit_example_plus.py \
  --gpt-ckpt "${GPT_CKPT}" \
  --vq-ckpt "${VQ_CKPT}" \
  --t5-path "${T5_PATH}" \
  --dataset "${DATASET_PATH}" \
  --log-dir "${LOG_DIR}" \
  --device "${DEVICE}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --cfg-scale "${CFG_SCALE}" \
  --gpt-model GPT-XL \
  --gpt-mode joint_cls_emb \
  --image-size 512 \
  --mixed-precision bf16 \
  --multi-cond \
  --add_ref_embed \
  --max_ref_num "${MAX_REF_NUM}" \
  --concat-target \
  --instruct-token-mode casual \
  --instruct-token-num "${INSTRUCT_TOKEN_NUM}" \
  --additional-info "${ADDITIONAL_INFO}"
