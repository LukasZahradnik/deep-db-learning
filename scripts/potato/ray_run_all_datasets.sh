#!/bin/bash

conda_env="deep-db-learning"

source "$(conda info --base)""/etc/profile.d/conda.sh"
conda activate "$conda_env"

run_timeout=$(expr 60 \* 60 \* 24) # 24 hours

MODEL_TYPE=$1
if [[ -z "$MODEL_TYPE" ]]; then
    MODEL_TYPE="transformer"
fi

EXPERIMENT_ID="all_datasets_${MODEL_TYPE}_$(date '+%d-%m-%Y_%H:%M:%S')"

too_small=(
    'cs' 'MuskSmall' 'nations' 'NBA' 'pubs' 'Pyrimidine' 'SAT' 'trains' 'university'
)

too_easy=(
    'Facebook' 'Airline' 'NCAA' 'AustralianFootball' 'world' 'Elti' 'geneea'
    'UTube' 'Dunur' 'cs' 'Basketball_women'
)

invalid_datasets=(
    'ccs' 'SFScores' 'Walmart' 'Countries'
)

CPU_CLS_DATASETS=(
    'Bupa' 'Carcinogenesis' 'Chess' 'CraftBeer'
    'Dallas' 'Dunur' 'financial' 'genes' 'Hepatitis_std' 'Mesh'
    'Mondial' 'medical' 'MuskSmall' 'mutagenesis' 'NCAA'
    'Pima' 'PremierLeague' 'PTE' 'Student_loan' 'Toxicology'
    'UW_std' 'WebKP'

    'CiteSeer' 'CORA' 'Credit' 'DCG' 'Hockey' 'imdb_MovieLens' 'Same_gen' 'voc'
)
GPU_CLS_DATASETS=(
    'ErgastF1' 'ftp' 'PubMed_Diabetes' 'SAP' 'tpcc' 'tpcds'
    
    'Accidents' 'imdb_ijs' 'legalActs' 'tpcd'
)
CPU_REG_DATASETS=(
    'Biodegradability' 'classicmodels' 'GOSales'
    'northwind' 'Triazine'
    
    'Basketball_men' 'restbase'
    
    'AdventureWorks2014' 'FNHK' 'lahman_2014' 'sakila' 'stats'
)
GPU_REG_DATASETS=(
    'Grants' 'tpch'
    
    'ConsumerExpenditures' 'employee' 'SalesDB' 'Seznam'
)

NUM_SAMPLES=16
EPOCHS=4000

# Create log directory
mkdir logs/${EXPERIMENT_ID}

# ******************************************
# Init Ray cluster
ip=$(hostname --ip-address)

port_head=7890
port_dashboard=7891

ray_address=$ip:$port_head
ray_dashboard_address=$ip:$port_dashboard

ray start --head --block --node-ip-address=$ip --port=$port_head \
 --dashboard-host=$ip --dashboard-port=$port_dashboard \
 --num-cpus=80 --num-gpus=4 --memory=300000000000 --object-store-memory=8000000000 \
 --log-style=record &> "logs/${EXPERIMENT_ID}/ray_head.log" &
ray_head=$!

echo "ray head address is ${ray_address}"
sleep 10
echo "ray dashboard available at http://${ray_dashboard_address}"

ray status --address=${ray_address}
# ******************************************

# Run experiment on different datasets
dataset_runs=()


for dataset in ${GPU_CLS_DATASETS[@]}; do
    mkdir logs/${EXPERIMENT_ID}/${dataset}
    python -u experiments/blueprint_mlflow.py --ray_address=${ray_address} \
    --log_dir=logs/${EXPERIMENT_ID}/${dataset} --experiment=pelesjak-deep-db-experiments-v2 \
    --run_name=${EXPERIMENT_ID}_${dataset} --model_type=${MODEL_TYPE} \
    --cuda --dataset=${dataset} --num_samples=${NUM_SAMPLES} --epochs=${EPOCHS} \
    --metric="acc" &> "logs/${EXPERIMENT_ID}/${dataset}/run.log" &
    dataset_runs+=($!)
    sleep 5
done

for dataset in ${GPU_REG_DATASETS[@]}; do
    mkdir logs/${EXPERIMENT_ID}/${dataset}
    python -u experiments/blueprint_mlflow.py --ray_address=${ray_address} \
    --log_dir=logs/${EXPERIMENT_ID}/${dataset} --experiment=pelesjak-deep-db-experiments-v2 \
    --run_name=${EXPERIMENT_ID}_${dataset} --model_type=${MODEL_TYPE} \
    --cuda --dataset=${dataset} --num_samples=${NUM_SAMPLES} --epochs=${EPOCHS} \
    --metric="nrmse" &> "logs/${EXPERIMENT_ID}/${dataset}/run.log" &
    dataset_runs+=($!)
    sleep 5
done

for dataset in ${CPU_CLS_DATASETS[@]}; do
    mkdir logs/${EXPERIMENT_ID}/${dataset}
    python -u experiments/blueprint_mlflow.py --ray_address=${ray_address} \
    --log_dir=logs/${EXPERIMENT_ID}/${dataset} --experiment=pelesjak-deep-db-experiments-v2 \
    --run_name=${EXPERIMENT_ID}_${dataset} --model_type=${MODEL_TYPE} \
    --dataset=${dataset} --num_samples=${NUM_SAMPLES} --epochs=${EPOCHS} \
    --metric="acc" &> "logs/${EXPERIMENT_ID}/${dataset}/run.log" &
    dataset_runs+=($!)
    sleep 5
done

for dataset in ${CPU_REG_DATASETS[@]}; do
    mkdir logs/${EXPERIMENT_ID}/${dataset}
    python -u experiments/blueprint_mlflow.py --ray_address=${ray_address} \
    --log_dir=logs/${EXPERIMENT_ID}/${dataset} --experiment=pelesjak-deep-db-experiments-v2 \
    --run_name=${EXPERIMENT_ID}_${dataset} --model_type=${MODEL_TYPE} \
    --dataset=${dataset} --num_samples=${NUM_SAMPLES} --epochs=${EPOCHS} \
    --metric="nrmse" &> "logs/${EXPERIMENT_ID}/${dataset}/run.log" &
    dataset_runs+=($!)
    sleep 5
done


# Stop after given timeout
function timeout_monitor() {
    echo "Run with stop in ${run_timeout}s"
    sleep "$run_timeout"
    ray stop
}

timeout_monitor &
timeout_monitor_pid=$!

# Wait for all experiments to finish
for run_pid in ${dataset_runs[@]}; do
    wait $run_pid
done

echo "All runs finished!"

# Stop ray cluster if was not stopped by timeout
if ps -p $timeout_monitor_pid > /dev/null
then
    kill $timeout_monitor_pid
    echo "Ray will stop in 30s"
    sleep 30
    ray stop
fi

kill $ray_head &

wait $ray_head

