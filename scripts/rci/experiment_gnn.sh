#!/bin/bash
#SBATCH --job-name=experiment-deep-db-gnn
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --time=16:00:00
#SBATCH --array=0-34

# datasets=('Accidents' 'Airline' 'Atherosclerosis' 'Basketball_women' 'Bupa' 'Carcinogenesis'
#     'Chess' 'CiteSeer' 'ConsumerExpenditures' 'CORA' 'CraftBeer' 'Credit' 'cs' 'Dallas' 'DCG' 'Dunur'
#     'Elti' 'ErgastF1' 'Facebook' 'financial' 'ftp' 'geneea' 'genes' 'Hepatitis_std' 'Hockey' 'imdb_ijs'
#     'imdb_MovieLens' 'KRK' 'legalActs' 'medical' 'Mondial' 'Mooney_Family' 'MuskSmall' 'mutagenesis'
#     'nations' 'NBA' 'NCAA' 'Pima' 'PremierLeague' 'PTE' 'PubMed_Diabetes' 'Same_gen' 'SAP' 'SAT'
#     'Shakespeare' 'Student_loan' 'Toxicology' 'tpcc' 'tpcd' 'tpcds' 'trains' 'university' 'UTube'
#     'UW_std' 'VisualGenome' 'voc' 'WebKP' 'world')
# datasets=('Dallas' 'world')

# datasets=('Accidents' 'AdventureWorks2014' 'Airline' 'Atherosclerosis' 'AustralianFootball' 'Basketball_men'
#     'Basketball_women' 'Biodegradability' 'Bupa' 'Carcinogenesis' 'ccs' 'CDESchools' 'Chess' 'CiteSeer'
#     'classicmodels' 'ConsumerExpenditures' 'CORA' 'Countries' 'CraftBeer' 'Credit' 'cs' 'Dallas' 'DCG'
#     'Dunur' 'Elti' 'employee' 'ErgastF1' 'Facebook' 'financial' 'FNHK' 'ftp' 'geneea' 'genes' 'GOSales'
#     'Grants' 'Hepatitis_std' 'Hockey' 'imdb_ijs' 'KRK' 'lahman_2014' 'legalActs' 'Mesh' 'Mondial'
#     'Mooney_Family' 'imdb_MovieLens' 'medical' 'MuskSmall' 'mutagenesis' 'nations' 'NBA' 'NCAA'
#     'northwind' 'Pima' 'PremierLeague' 'PTC' 'PTE' 'PubMed_Diabetes' 'pubs' 'Pyrimidine' 'restbase'
#     'sakila' 'SalesDB' 'Same_gen' 'SAP' 'SAT' 'Seznam' 'SFScores' 'Shakespeare' 'stats' 'Student_loan'
#     'Toxicology' 'tpcc' 'tpcd' 'tpcds' 'tpch' 'trains' 'Triazine' 'university' 'UTube' 'UW_std' 'VisualGenome'
#     'voc' 'Walmart' 'WebKP' 'world'
# )

# cls_datasets=('Accidents' 'Airline' 'Atherosclerosis' 'AustralianFootball' 
#     'Basketball_women' 'Bupa' 'Carcinogenesis' 'Chess' 'CiteSeer' 'CORA'
#     'CraftBeer' 'Credit' 'cs' 'Dallas' 'DCG' 'Dunur' 'Elti' 'ErgastF1' 
#     'Facebook' 'financial' 'ftp' 'geneea' 'genes' 'Hepatitis_std' 'Hockey' 
#     'imdb_ijs' 'KRK' 'legalActs' 'Mesh' 'Mondial' 'imdb_MovieLens' 'medical'
#     'MuskSmall' 'mutagenesis' 'nations' 'NBA' 'NCAA' 'Pima' 'PremierLeague'
#     'PTE' 'PubMed_Diabetes' 'Same_gen' 'SAP' 'SAT' 'Student_loan' 'Toxicology' 
#     'tpcc' 'tpcd' 'tpcds' 'trains' 'university' 'UTube' 'UW_std' 'voc' 'WebKP' 'world'
# )

# reg_datasets=('AdventureWorks2014' 'Basketball_men' 'Biodegradability' 'ccs'
#     'CDESchools' 'classicmodels' 'ConsumerExpenditures' 'Countries' 'employee'
#     'FNHK' 'GOSales' 'Grants' 'lahman_2014' 'northwind' 'PTC' 'pubs' 'Pyrimidine'
#     'restbase' 'sakila' 'SalesDB' 'Seznam' 'SFScores' 'stats' 'tpch' 'Triazine'
#     'Walmart'
# )

paper_datasets=(
    'Accidents' 'AdventureWorks2014' 'Basketball_men' 'Biodegradability' 'Carcinogenesis'
    'classicmodels' 'ConsumerExpenditures' 'CraftBeer' 'Dallas' 'DCG'
    'employee' 'financial' 'FNHK' 'GOSales' 'Grants'
    'imdb_ijs' 'Mondial' 'MuskSmall' 'mutagenesis' 'northwind' 
    'Pima' 'PremierLeague' 'PubMed_Diabetes' 'restbase' 'sakila' 
    'SalesDB' 'Same_gen' 'Seznam' 'stats'  'Toxicology' 
    'tpcd' 'Triazine' 'UW_std' 'voc' 'WebKP'
)

id="${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}"

dataset=${paper_datasets[$SLURM_ARRAY_TASK_ID]}

conda_env="relational-py"

source "$(conda info --base)""/etc/profile.d/conda.sh"
conda activate "$conda_env"

run_dir=logs/${id}
mkdir -p ${run_dir}

log_dir=${run_dir}/${dataset}
mkdir -p ${log_dir}


python -u experiments/blueprint_mlflow.py --ray_address="local" --dataset=${dataset} \
    --experiment="pelesjak-deep-db-experiments-v3" --run_name=${id} --log_dir=${log_dir} \
    --seed=42 --num_samples=24 --model_type="honza" --num_cpus=${SLURM_CPUS_PER_TASK} &> "${log_dir}/run.log"



