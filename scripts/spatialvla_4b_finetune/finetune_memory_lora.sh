set -x

# Minimal FIFO-memory LoRA fine-tuning for SpatialVLA.
# Override variables from the shell, e.g.:
#   mixture=libero_spatial_no_noops MEMORY_TRAIN_WINDOW=8 bash scripts/spatialvla_4b_finetune/finetune_memory_lora.sh

DEBUG=${DEBUG:-true}
if [ "$DEBUG" = true ]; then
  GPUS=${GPUS:-1}
  GPUS_PER_NODE=${GPUS_PER_NODE:-1}
  PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
  shuffle_buffer_size=${shuffle_buffer_size:-2}
  NUM_WORKERS=${NUM_WORKERS:-0}
  TORCH_RUN_ARGS=${TORCH_RUN_ARGS:-"--standalone --nnodes=1"}
  save_steps=${save_steps:-50}
fi

GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NODES=$((GPUS / GPUS_PER_NODE))
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-4}
BATCH_SIZE=${BATCH_SIZE:-$((GPUS * PER_DEVICE_BATCH_SIZE))}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))

model_name_or_path=${model_name_or_path:-../pretrained/spatialvla-4b-224-pt}
data_root_dir=${data_root_dir:-/oss/vla_ptm_hwfile/DATA/open_x_embodiment_converted}
mixture=${mixture:-libero_10_no_noops}
NUM_WORKERS=${NUM_WORKERS:-1}
shuffle_buffer_size=${shuffle_buffer_size:-8192}

lr=${lr:-5e-4}
lora=${lora:-32}
lora_alpha=${lora_alpha:-32}
lora_target=${lora_target:-llm_linear}
epoch=${epoch:-50}
save_steps=${save_steps:-10000}

MEMORY_TRAIN_WINDOW=${MEMORY_TRAIN_WINDOW:-8}
MEMORY_WRITE_TOKENS=${MEMORY_WRITE_TOKENS:-4}
MEMORY_BANK_SIZE=${MEMORY_BANK_SIZE:-64}
MEMORY_RETRIEVE_TOKENS=${MEMORY_RETRIEVE_TOKENS:-8}
MEMORY_NUM_HEADS=${MEMORY_NUM_HEADS:-8}

cur_time=$(date "+%H-%M-%S")
date_dir=$(date "+%Y-%m-%d")
note=$(basename $model_name_or_path)_memT${MEMORY_TRAIN_WINDOW}_bank${MEMORY_BANK_SIZE}_write${MEMORY_WRITE_TOKENS}_ret${MEMORY_RETRIEVE_TOKENS}_lr${lr}_bs${PER_DEVICE_BATCH_SIZE}_gpu${GPUS}_r${lora}_a${lora_alpha}
OUTPUT_DIR=${resume_path:-outputs/spatialvla_4b_memory_lora/$date_dir/${cur_time}_${mixture}_${note}}
mkdir -p $OUTPUT_DIR

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER="pytorch"
TORCH_RUN_ARGS=${TORCH_RUN_ARGS:-"--nnodes $NODES --nproc-per-node $GPUS_PER_NODE --master_addr $MASTER_ADDR --master_port $MASTER_PORT"}

cp $(realpath "$0") ${OUTPUT_DIR}

torchrun $TORCH_RUN_ARGS \
  train/spatialvla_finetune.py \
  --model_name_or_path ${model_name_or_path} \
  --use_memory True \
  --memory_train_window ${MEMORY_TRAIN_WINDOW} \
  --memory_write_tokens ${MEMORY_WRITE_TOKENS} \
  --memory_bank_size ${MEMORY_BANK_SIZE} \
  --memory_retrieve_tokens ${MEMORY_RETRIEVE_TOKENS} \
  --memory_num_heads ${MEMORY_NUM_HEADS} \
  --lora ${lora} \
  --lora_alpha ${lora_alpha} \
  --lora_target ${lora_target} \
  --ignore_data_skip True \
  --data_root_dir ${data_root_dir} \
  --data_mix ${mixture} \
  --shuffle_buffer_size ${shuffle_buffer_size} \
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
  --logging_steps 500 \
  --do_train True \
  --grad_checkpoint True \
  --deepspeed scripts/zero1.json \
  --report_to tensorboard \
  --log_level warning
