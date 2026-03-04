#!/bin/bash

set -e

# Dataset and output naming.
dataset="/home/tione/notebook/yingcaihe/sunhaiyue/SpatialSubject200K/unzipSubject200K_test"

# Common model paths.
VQ_CKPT="/home/tione/notebook/yingcaihe/sunhaiyue/ckpt/llamagen_t2i/vq_ds16_t2i.pt"
T5_PATH="/home/tione/notebook/yingcaihe/sunhaiyue/ckpt"
LOG_DIR="/home/tione/notebook/yingcaihe/sunhaiyue/TokenAR2/checkpoints/eval"
CKPT_DIR="/home/tione/notebook/yingcaihe/sunhaiyue/TokenAR2/checkpoints/train/BaseModel/checkpoints"

# Sampling config.
CFG_SCALE=3
SEED=83
MAX_REF_NUM=2
INSTRUCT_TOKEN_NUM=120
BATCH_SIZE=8
NUM_WORKERS=4
MIXED_PRECISION="bf16"

mapfile -t ckpts < <(find "${CKPT_DIR}" -maxdepth 1 -type f -name "*.pt" | sort)

if [ ${#ckpts[@]} -eq 0 ]; then
    echo "No checkpoint files found in ${CKPT_DIR}"
    exit 1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    mapfile -t gpu_ids < <(nvidia-smi --query-gpu=index --format=csv,noheader)
else
    gpu_ids=(0)
fi

if [ ${#gpu_ids[@]} -eq 0 ]; then
    echo "No GPU detected."
    exit 1
fi

echo "Detected GPUs: ${gpu_ids[*]}"
echo "Each GPU will run at most one checkpoint at a time."

wait_for_free_gpu() {
    while true; do
        for gpu_index in "${!gpu_ids[@]}"; do
            pid="${gpu_pids[$gpu_index]}"
            if [ -z "${pid}" ]; then
                echo "${gpu_index}"
                return 0
            fi
            if ! kill -0 "${pid}" 2>/dev/null; then
                wait "${pid}" || true
                gpu_pids[$gpu_index]=""
                echo "${gpu_index}"
                return 0
            fi
        done
        sleep 5
    done
}

gpu_pids=()
for gpu_index in "${!gpu_ids[@]}"; do
    gpu_pids[$gpu_index]=""
done

for ckpt in "${ckpts[@]}"; do
    ckpt_parent_name="$(basename "$(dirname "${CKPT_DIR}")")"
    ckpt_stem="$(basename "${ckpt}" .pt)"
    additional_info="${ckpt_parent_name}_${ckpt_stem}"

    free_gpu_index="$(wait_for_free_gpu)"
    gpu_id="${gpu_ids[$free_gpu_index]}"

    echo "Processing checkpoint: ${ckpt}"
    echo "Launching on cuda:${gpu_id} with batch size ${BATCH_SIZE}"

    CUDA_VISIBLE_DEVICES="${gpu_id}" python3 autoregressive/sample/sample_edit_example_plus.py \
        --gpt-ckpt "${ckpt}" \
        --vq-ckpt "${VQ_CKPT}" \
        --t5-path "${T5_PATH}" \
        --log-dir "${LOG_DIR}" \
        --add_ref_embed \
        --multi-cond \
        --cfg-scale "${CFG_SCALE}" \
        --seed "${SEED}" \
        --max_ref_num "${MAX_REF_NUM}" \
        --additional-info "${additional_info}" \
        --device "cuda:0" \
        --dataset "${dataset}" \
        --batch-size "${BATCH_SIZE}" \
        --num-workers "${NUM_WORKERS}" \
        --instruct-token-mode casual \
        --instruct-token-num "${INSTRUCT_TOKEN_NUM}" \
        --mixed-precision "${MIXED_PRECISION}" &

    gpu_pids[$free_gpu_index]=$!
    echo "--------------------------------------------------"
done

for pid in "${gpu_pids[@]}"; do
    if [ -n "${pid}" ]; then
        wait "${pid}"
    fi
done

echo "All checkpoints have been processed."
