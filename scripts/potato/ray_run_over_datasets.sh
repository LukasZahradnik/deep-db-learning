#!/bin/bash

run_timeout=$(expr 60 \* 60 \* 24) # 24 hours
# run_timeout=$(expr 60 ) # 24 hours

conda_env="deep-db-learning"

datasets=('Accidents' 'AdventureWorks2014' 'Airline' 'Atherosclerosis' 'AustralianFootball' 'Basketball_men'
    'Basketball_women' 'Biodegradability' 'Bupa' 'Carcinogenesis' 'ccs' 'CDESchools' 'Chess' 'CiteSeer'
    'classicmodels' 'ConsumerExpenditures' 'CORA' 'Countries' 'CraftBeer' 'Credit' 'cs' 'Dallas' 'DCG'
    'Dunur' 'Elti' 'employee' 'ErgastF1' 'Facebook' 'financial' 'FNHK' 'ftp' 'geneea' 'genes' 'GOSales'
    'Grants' 'Hepatitis_std' 'Hockey' 'imdb_ijs' 'KRK' 'lahman_2014' 'legalActs' 'Mesh' 'Mondial'
    'Mooney_Family' 'imdb_MovieLens' 'medical' 'MuskSmall' 'mutagenesis' 'nations' 'NBA' 'NCAA'
    'northwind' 'Pima' 'PremierLeague' 'PTC' 'PTE' 'PubMed_Diabetes' 'pubs' 'Pyrimidine' 'restbase'
    'sakila' 'SalesDB' 'Same_gen' 'SAP' 'SAT' 'Seznam' 'SFScores' 'Shakespeare' 'stats' 'Student_loan'
    'Toxicology' 'tpcc' 'tpcd' 'tpcds' 'tpch' 'trains' 'Triazine' 'university' 'UTube' 'UW_std' 'VisualGenome'
    'voc' 'Walmart' 'WebKP' 'world'
)

cls_datasets=('Accidents' 'Airline' 'Atherosclerosis' 'AustralianFootball' 
    'Basketball_women' 'Bupa' 'Carcinogenesis' 'Chess' 'CiteSeer' 'CORA'
    'CraftBeer' 'Credit' 'cs' 'Dallas' 'DCG' 'Dunur' 'Elti' 'ErgastF1' 
    'Facebook' 'financial' 'ftp' 'geneea' 'genes' 'Hepatitis_std' 'Hockey' 
    'imdb_ijs' 'KRK' 'legalActs' 'Mesh' 'Mondial' 'imdb_MovieLens' 'medical'
    'MuskSmall' 'mutagenesis' 'nations' 'NBA' 'NCAA' 'Pima' 'PremierLeague'
    'PTE' 'PubMed_Diabetes' 'Same_gen' 'SAP' 'SAT' 'Student_loan' 'Toxicology' 
    'tpcc' 'tpcd' 'tpcds' 'trains' 'university' 'UTube' 'UW_std' 'voc' 'WebKP' 'world'
)

reg_datasets=('AdventureWorks2014' 'Basketball_men' 'Biodegradability' 'ccs'
    'CDESchools' 'classicmodels' 'ConsumerExpenditures' 'Countries' 'employee'
    'FNHK' 'GOSales' 'Grants' 'lahman_2014' 'northwind' 'PTC' 'pubs' 'Pyrimidine'
    'restbase' 'sakila' 'SalesDB' 'Seznam' 'SFScores' 'stats' 'tpch' 'Triazine'
    'Walmart'
)

try_datasets=('Chess' 'CORA' 'mutagenesis')

NAME=$1
if [[ -z "$NAME" ]]; then
    NAME="benchmark"
fi

id="${NAME}_$(date '+%d-%m-%Y_%H:%M:%S')"

mkdir logs/$id

ip=$(hostname --ip-address)

port_head=7890
port_dashboard=7891

ip_head=$ip:$port_head
ip_dashboard=$ip:$port_dashboard

ray start --head --block --node-ip-address=$ip --port=$port_head \
 --dashboard-host=$ip --dashboard-port=$port_dashboard \
 --num-cpus=32 --num-gpus=4 --memory=64000000000 --object-store-memory=8000000000 \
 --log-style=record &> "logs/$id/ray_head.log" &
ray_head=$!

echo "ray head address is $ip_head"
sleep 10
echo "ray dashboard available at http://$ip_dashboard"


ray status --address=$ip_head

download_runs=()
for d in ${cls_datasets[@]}; do
    mkdir logs/$id/$d
    python -u scripts/download_dataset.py --dataset=$d &> "logs/$id/$d/download.log" &
    download_runs+=($!)
    sleep 5
done

for run_pid in ${download_runs[@]}; do
    wait $run_pid
done

dataset_runs=()
for d in ${cls_datasets[@]}; do
    python -u experiments/blueprint_mlflow.py --ray_address=$ip_head \
    --experiment=pelesjak-deep-db-experiments-v2-honza --log_dir=logs/$id/$d --run_name=${id}_${d} \
    --cuda --dataset=$d --num_samples=8 &> "logs/$id/$d/run.log" &
    dataset_runs+=($!)
done

function timeout_monitor() {
    echo "Run with stop in ${run_timeout}s"
    sleep "$run_timeout"
    ray stop
}

timeout_monitor &
timeout_monitor_pid=$!

for run_pid in ${dataset_runs[@]}; do
    wait $run_pid
done

echo "All runs finished!"

if ps -p $timeout_monitor_pid > /dev/null
then
    kill $timeout_monitor_pid
    echo "Ray will stop in 30s"
    sleep 30
    ray stop
fi

kill $ray_head&

wait $ray_head

