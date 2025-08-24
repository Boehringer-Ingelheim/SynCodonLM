#!/bin/bash
#SBATCH --job-name=bert
#SBATCH --partition=gpu
#SBATCH --nodes=3
#SBATCH --gpus-per-node=4
#SBATCH --nodelist=your nodes
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --output=output_%j.txt
#SBATCH --error=error_%j.txt

unset NCCL_DEBUG

export NCCL_SOCKET_IFNAME=$(ip -o link show | awk -F': ' '{print $2}' | grep mlx | head -n 1)
echo "Using NCCL_SOCKET_IFNAME: $NCCL_SOCKET_IFNAME"

export NCCL_IB_DISABLE=0

export NCCL_NET=IB
export NCCL_P2P_LEVEL=SYS

export NCCL_NET_GDR_LEVEL=2
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=3600
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

export TORCH_USE_CUDA_DSA=1

export MASTER_ADDR=$(getent hosts $(scontrol show hostname $SLURM_NODELIST | head -n 1) | awk '{ print $1 }')
export MASTER_PORT=60000
if ss -tuln | grep ":60000"; then
    echo "Port 60000 is already in use!"
    exit 1
fi

export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "RANK: $RANK"
echo "LOCAL_RANK: $LOCAL_RANK"
echo "Allocated nodes:"
scontrol show hostname $SLURM_NODELIST



echo "Activating virtual environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env



echo "Running Python script with srun..."
srun stdbuf -oL python $(pwd)/pretrain.py
