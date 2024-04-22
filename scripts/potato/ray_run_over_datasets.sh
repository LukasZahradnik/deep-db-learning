#!/bin/bash

conda_env="deep-db-learning"

source "$(conda info --base)""/etc/profile.d/conda.sh"
conda activate "$conda_env"

run_timeout=$(expr 60 \* 60 \* 24) # 24 hours

cls_datasets=('Accidents' 'Airline' 'Atherosclerosis' 'AustralianFootball' 
    'Basketball_women' 'Bupa' 'Carcinogenesis' 'Chess' 'CiteSeer' 'CORA'
    'CraftBeer' 'Credit' 'cs' 'Dallas' 'DCG' 'Dunur' 'Elti' 'ErgastF1' 
    'Facebook' 'financial' 'ftp' 'geneea' 'genes' 'Hepatitis_std' 'Hockey' 
    'imdb_ijs' 'KRK' 'legalActs' 'Mesh' 'Mondial' 'imdb_MovieLens' 'medical'
    'MuskSmall' 'mutagenesis' 'nations' 'NBA' 'NCAA' 'Pima' 'PremierLeague'
    'PTE' 'PubMed_Diabetes' 'Same_gen' 'SAP' 'SAT' 'Student_loan' 'Toxicology' 
    'tpcc' 'tpcd' 'tpcds' 'trains' 'university' 'UTube' 'UW_std' 'voc' 'WebKP' 'world'
)

cls_orig_datasets=('Atherosclerosis' 'Carcinogenesis' 'Chess' 'CiteSeer'
    'cs' 'DCG' 'Dunur' 'MuskSmall' 'mutagenesis' 'NCAA' 'Pima' 'PTE' 
    'Toxicology' 'university' 'UTube' 'WebKP'
)

cls_xs_datasets=('Basketball_women' 'Bupa' 'Carcinogenesis' 'Chess' 'CraftBeer' 'cs'
    'Dallas' 'Dunur' 'Facebook' 'financial' 'genes' 'Hepatitis_std' 'Mesh'
    'Mondial' 'medical' 'MuskSmall' 'mutagenesis' 'nations' 'NBA' 'NCAA'
    'Pima' 'PremierLeague' 'PTE' 'SAT' 'Student_loan' 'Toxicology' 'trains'
    'university' 'UTube' 'UW_std' 'WebKP' 'world'
)

cls_s_datasets=('AustralianFootball' 'CiteSeer' 'CORA' 'Credit' 'DCG' 'Elti' 'geneea'
    'Hockey' 'imdb_MovieLens' 'Same_gen' 'voc'
)

cls_m_datasets=('ErgastF1' 'ftp' 'PubMed_Diabetes' 'SAP' 'tpcc' 'tpcds')

cls_l_datasets=('Accidents' 'Airline' 'imdb_ijs' 'legalActs' 'tpcd')

reg_datasets=('AdventureWorks2014' 'Basketball_men' 'Biodegradability' 'ccs'
    'classicmodels' 'ConsumerExpenditures' 'Countries' 'employee'
    'FNHK' 'GOSales' 'Grants' 'lahman_2014' 'northwind' 
    'pubs' 'Pyrimidine' 'restbase' 'sakila' 'SalesDB' 'Seznam' 'SFScores' 'stats' 
    'tpch' 'Triazine' 'Walmart'
)

reg_xs_datasets=('Biodegradability' 'ccs' 'classicmodels' 'Countries' 'GOSales'
    'northwind' 'pubs' 'Pyrimidine' 'Triazine'
)

reg_s_datasets=('Basketball_men' 'restbase')

reg_m_datasets=('AdventureWorks2014' 'FNHK' 'lahman_2014' 'sakila' 'SFScores' 'stats')

reg_l_datasets=('Grants' 'tpch')

reg_xl_datasets=('ConsumerExpenditures' 'employee' 'SalesDB' 'Seznam' 'Walmart')

DATASET_NAME=$1
if [[ -z "$DATASET_NAME" ]]; then
    DATASET_NAME="cls_orig_datasets"
fi

MODEL_TYPE=$2
if [[ -z "$MODEL_TYPE" ]]; then
    MODEL_TYPE="transformer"
fi

EXPERIMENT_ID="${DATASET_NAME}_${MODEL_TYPE}_$(date '+%d-%m-%Y_%H:%M:%S')"

datasets_exp=${DATASET_NAME}[@]
DATASETS=${!datasets_exp}

NUM_SAMPLES=4
EPOCHS=2000
METRIC="acc"
if [[ $DATASET_NAME == *"reg"* ]]; then
  METRIC="mse"
fi

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
 --num-cpus=32 --num-gpus=4 --memory=64000000000 --object-store-memory=8000000000 \
 --log-style=record &> "logs/${EXPERIMENT_ID}/ray_head.log" &
ray_head=$!

echo "ray head address is ${ray_address}"
sleep 10
echo "ray dashboard available at http://${ray_dashboard_address}"

ray status --address=${ray_address}
# ******************************************

# download_runs=()
# for d in ${DATASETS[@]}; do
#     mkdir logs/${EXPERIMENT_ID}/$d
#     python -u scripts/download_dataset.py --dataset=$d &> "logs/${EXPERIMENT_ID}/$d/download.log" &
#     download_runs+=($!)
#     sleep 5
# done

# for run_pid in ${download_runs[@]}; do
#     wait $run_pid
# done

# Run experiment on different datasets
dataset_runs=()
for dataset in ${DATASETS[@]}; do
    mkdir logs/${EXPERIMENT_ID}/${dataset}
    python -u experiments/blueprint_mlflow.py --ray_address=${ray_address} \
    --log_dir=logs/${EXPERIMENT_ID}/${dataset} --experiment=pelesjak-deep-db-experiments-v2 \
    --run_name=${EXPERIMENT_ID}_${dataset} --model_type=${MODEL_TYPE} \
    --cuda --dataset=${dataset} --num_samples=${NUM_SAMPLES} --epochs=${EPOCHS} \
    --metric=${METRIC} &> "logs/${EXPERIMENT_ID}/${dataset}/run.log" &
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

kill $ray_head&

wait $ray_head

