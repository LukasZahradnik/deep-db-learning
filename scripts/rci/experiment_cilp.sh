#!/bin/bash
#SBATCH --job-name=experiment-deep-db-cilp
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --time=20:00:00
#SBATCH --ntasks=1


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

start_index=$1
end_index=$2

paper_datasets=(
    'AdventureWorks2014' 'Basketball_men' 'Biodegradability' 'Carcinogenesis'
    'classicmodels' 'ConsumerExpenditures' 'CraftBeer' 'Dallas' 'DCG'
    'financial' 'FNHK' 'imdb_ijs' 'Mondial' 'MuskSmall' 'mutagenesis' 'northwind' 
    'PremierLeague' 'PubMed_Diabetes' 'restbase'  
    'Same_gen' 'Seznam' 'stats'  'Toxicology' 
    'Triazine' 'UW_std' 'WebKP'
)
datasets=("${paper_datasets[@]:${start_index}:${end_index}}")

# Accidents, employee, Grants, tpcd, SalesDB removed due to memory issues
# Pima, 'voc' no fact column
# GOSales, sakila incompatible with getml
conda_env="relational-py"

source "$(conda info --base)""/etc/profile.d/conda.sh"
conda activate "$conda_env"

id="${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
run_dir=logs/${id}
mkdir -p ${run_dir}

i=1
for dataset in "${datasets[@]}"
do
    log_dir=${run_dir}/${dataset}
    mkdir -p ${log_dir}

    python -u experiments/getml_cilp.py --dataset=${dataset} --experiment="pelesjak-deep-db-experiments-v3" \
        --run_name=${id} --log_dir=${log_dir} --seed=42 &> "${log_dir}/run.log"

    ((i++))
done


