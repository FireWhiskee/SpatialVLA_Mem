set -x

# Minimal FIFO-memory LoRA fine-tuning for SpatialVLA.
# Override variables from the shell, e.g.:
#   mixture=libero_spatial_no_noops MEMORY_TRAIN_WINDOW=8 bash scripts/spatialvla_4b_finetune/finetune_memory_lora.sh

DEBUG=${DEBUG:-false}
if [ "$DEBUG" = true ]; then
  GPUS=1
  GPUS_PER_NODE=1
  PER_DEVICE_BATCH_SIZE=2
  shuffle_buffer_size=2
  mixture=bridge_orig
  NUM_WORKERS=0
  TORCH_RUN_ARGS="--standalone --nnodes=1"
  save_steps=50
fi

GPUS=${GPUS:-4}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
NODES=$((GPUS / GPUS_PER_NODE))
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
BATCH_SIZE=${BATCH_SIZE:-16}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))

mixture=${mixture:-libero_10_no_noops}
NUM_WORKERS=${NUM_WORKERS:-1}
shuffle_buffer_size=${shuffle_buffer_size:-8192} # large buffer for better shuffling, we use 131072 in pretrain

lr=${lr:-2e-4}
lora=${lora:-16}
lora_alpha=${lora_alpha:-16}
lora_target="linear"

epoch=${epoch:-2}
save_steps=${save_steps:-100}

MEMORY_TRAIN_WINDOW=${MEMORY_TRAIN_WINDOW:-8}
MEMORY_WRITE_TOKENS=${MEMORY_WRITE_TOKENS:-4}
MEMORY_BANK_SIZE=${MEMORY_BANK_SIZE:-64}
MEMORY_RETRIEVE_TOKENS=${MEMORY_RETRIEVE_TOKENS:-8}
MEMORY_NUM_HEADS=${MEMORY_NUM_HEADS:-8}
MEMORY_ALPHA_INIT=${MEMORY_ALPHA_INIT:-0.05}
MEMORY_DETACH_WRITE=${MEMORY_DETACH_WRITE:-False}
FIX_RAW_LENGTH=${FIX_RAW_LENGTH:-2000}
MAX_STEPS=${MAX_STEPS:-}

DATA_LIMIT_ARGS=""
if [ -n "$FIX_RAW_LENGTH" ]; then
  DATA_LIMIT_ARGS="${DATA_LIMIT_ARGS} --fix_raw_length ${FIX_RAW_LENGTH}"
fi
if [ -n "$MAX_STEPS" ]; then
  DATA_LIMIT_ARGS="${DATA_LIMIT_ARGS} --max_steps ${MAX_STEPS}"
fi

cur_time=$(date "+%H-%M-%S")
date_dir=$(date "+%Y-%m-%d")

# resume training from ckpt
model_name_or_path=/root/autodl-tmp/hf/pretrained/spatialvla-4b-224-pt
data_root_dir=/root/autodl-tmp/data/modified_libero_rlds
note=$(basename $model_name_or_path)_memT${MEMORY_TRAIN_WINDOW}_bank${MEMORY_BANK_SIZE}_write${MEMORY_WRITE_TOKENS}_ret${MEMORY_RETRIEVE_TOKENS}_alpha${MEMORY_ALPHA_INIT}_detach${MEMORY_DETACH_WRITE}_lr${lr}_bs${PER_DEVICE_BATCH_SIZE}_node$((GPUS / GPUS_PER_NODE))_gpu${GPUS}_r${lora}_a${lora_alpha}_ep${epoch}_${lora_target}
OUTPUT_DIR=${resume_path:-/root/autodl-tmp/outputs/spatialvla_4b_memory_lora/$date_dir/${cur_time}_${mixture}_${note}}
mkdir -p $OUTPUT_DIR

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export TF_CPP_MIN_LOG_LEVEL=3
# export LD_PRELOAD=../libtcmalloc.so.4.5.3 # optional, for better memory management
# export TRITON_CACHE_DIR=~/.triton

cp $(realpath "$0") ${OUTPUT_DIR}

export LAUNCHER="pytorch"
TORCH_RUN_ARGS=${TORCH_RUN_ARGS:-"--nnodes $NODES --nproc-per-node $GPUS_PER_NODE"}

torchrun $TORCH_RUN_ARGS \
  train/spatialvla_finetune.py \
  --model_name_or_path ${model_name_or_path} \
  ${ADAPT_ARGS} \
  --use_memory True \
  --memory_train_window ${MEMORY_TRAIN_WINDOW} \
  --memory_write_tokens ${MEMORY_WRITE_TOKENS} \
  --memory_bank_size ${MEMORY_BANK_SIZE} \
  --memory_retrieve_tokens ${MEMORY_RETRIEVE_TOKENS} \
  --memory_num_heads ${MEMORY_NUM_HEADS} \
  --memory_alpha_init ${MEMORY_ALPHA_INIT} \
  --memory_detach_write ${MEMORY_DETACH_WRITE} \
  --lora ${lora} \
  --lora_alpha ${lora_alpha} \
  --lora_target ${lora_target} \
  --ignore_data_skip True \
  --data_root_dir ${data_root_dir} \
  --data_mix ${mixture} \
  --shuffle_buffer_size ${shuffle_buffer_size} \
  ${DATA_LIMIT_ARGS} \
  --obs_backward_steps $((MEMORY_TRAIN_WINDOW - 1)) \
  --obs_backward_delta 1 \
  --action_forward_steps 0 \
  --flash_attn True \
  --output_dir ${OUTPUT_DIR} \
  --overwrite_output_dir False \
  --freeze_vision_tower True \
  --dataloader_num_workers ${NUM_WORKERS} \
  --bf16 True \
  --tf32 True \
  --num_train_epochs ${epoch} \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --save_strategy steps \
  --save_steps ${save_steps} \
  --save_total_limit 3 \
  --learning_rate ${lr} \
  --weight_decay 0.0 \
  --warmup_ratio 0.005 \
  --lr_scheduler_type linear \
  --logging_steps 20 \
  --do_train True \
  --grad_checkpoint True \
  --deepspeed scripts/zero1.json \
  --report_to tensorboard \
  --log_level warning
