export CUDA_VISIBLE_DEVICES=0,1,2
export NCCL_P2P_LEVEL=NVL
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1200

nohup accelerate launch \
  --num_processes 3 \
  --gpu_ids 0,1,2 \
  --config_file configs/deepspeed/zero3_bf16.yaml \
  scripts/train_accel.py > logs/train_qwen3_0.6b.log 2>&1 &