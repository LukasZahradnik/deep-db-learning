#!/bin/bash

conda_env="deep-db-learning"

source "$(conda info --base)""/etc/profile.d/conda.sh"
conda activate "$conda_env"

run_timeout=$(expr 60 \* 60 \* 24) # 24 hours

MODEL_TYPE=$1
if [[ -z "$MODEL_TYPE" ]]; then
    MODEL_TYPE="getml_xgboost"
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

ALL_DATASETS=(
    # 'Bupa' 'Carcinogenesis' 'Chess' 'CraftBeer'
    # 'Dallas' 'Dunur' 'financial' 'genes' 'Hepatitis_std' 'Mesh'
    # 'Mondial' 'medical' 'MuskSmall' 'mutagenesis' 'NCAA'
    # 'Pima' 
    'PremierLeague' 'PTE' 'Student_loan' 'Toxicology'
    'UW_std' 'WebKP'

    'CiteSeer' 'CORA' 'Credit' 'DCG' 'Hockey' 'imdb_MovieLens' 'Same_gen' 'voc'
    
    'ErgastF1' 'ftp' 'PubMed_Diabetes' 'SAP' 'tpcc' 'tpcds'
    
    'Accidents' 'imdb_ijs' 'legalActs' 'tpcd'
    
    'Biodegradability' 'classicmodels' 'GOSales'
    'northwind' 'Triazine'
    
    'Basketball_men' 'restbase'
    
    'AdventureWorks2014' 'FNHK' 'lahman_2014' 'sakila' 'stats'
    
    'Grants' 'tpch'
    
    'ConsumerExpenditures' 'employee' 'SalesDB' 'Seznam'
)

# Create log directory
mkdir logs/${EXPERIMENT_ID}

# ******************************************

# Stop after given timeout
function timeout_monitor() {
    echo "Run with stop in ${run_timeout}s"
    sleep "$run_timeout"
    python -c """
    import getml
    getml.engine.shutdown()
    """
}

timeout_monitor &
timeout_monitor_pid=$!

# Run experiment on different datasets
for dataset in ${ALL_DATASETS[@]}; do
    mkdir logs/${EXPERIMENT_ID}/${dataset}
    python -u experiments/getml_xgboost.py --dataset=${dataset} \
    --experiment=pelesjak-deep-db-experiments-v2 --log_dir=logs/${EXPERIMENT_ID}/${dataset} \
    --run_name=${EXPERIMENT_ID}_${dataset} &> "logs/${EXPERIMENT_ID}/${dataset}/run.log"
done


echo "All runs finished!"

python -c """
import getml
getml.engine.shutdown()
"""
