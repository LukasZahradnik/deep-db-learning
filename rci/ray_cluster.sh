#!/bin/bash

USE_CUDA=false

NAME=$1
if [[ -z "$NAME" ]]; then
    NAME="benchmark"
fi

PYTHON_SCRIPT=$2
if [[ -z "$PYTHON_SCRIPT" ]]; then
    PYTHON_SCRIPT="experiments/blueprint_mlflow.py"
fi

id="${NAME}_$(date '+%d-%m-%Y_%H:%M:%S')_$(openssl rand -hex 4)"

if [ "$USE_CUDA" = true ] ; then
    echo "Using CUDA"
    sbatch -o logs/$id/batch.log rci/ray_cluster_cuda.batch $id ${PYTHON_SCRIPT}
else
    echo "Not using CUDA"
    sbatch -o logs/$id/batch.log rci/ray_cluster.batch $id ${PYTHON_SCRIPT}
fi

wait