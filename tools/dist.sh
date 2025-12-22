#!/usr/bin/env bash

set -x

FILE=$1
CONFIG=$2
GPUS=$3
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-$((28500 + $RANDOM % 2000))}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
DEEPSPEED=${DEEPSPEED:-deepspeed_zero2}

# torchrun --nnodes $NNODES 
# --nproc_per_node $GPUS_PER_NODE 
# --node_rank $SLURM_NODEID 
# --master_addr $(scontrol show hostname $SLURM_NODELIST | head -n1) 
# --master_port ${MASTER_PORT} 
if command -v torchrun &> /dev/null
then
  echo "Using torchrun mode."
  PYTHONPATH="$(dirname $0)/..":$PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    torchrun --nnodes=${NNODES} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${PORT} \
    --nproc_per_node=${GPUS} \
    tools/${FILE}.py ${CONFIG} --launcher pytorch --deepspeed $DEEPSPEED "${@:4}"
else
  echo "Using launch mode."
  PYTHONPATH="$(dirname $0)/..":$PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python -m torch.distributed.launch \
    --nnodes=${NNODES} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${PORT} \
    --nproc_per_node=${GPUS} \
    tools/${FILE}.py ${CONFIG} --launcher pytorch --deepspeed $DEEPSPEED "${@:4}"
fi
