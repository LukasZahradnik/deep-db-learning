#!/bin/bash

# datasets=('Accidents' 'Airline' 'Atherosclerosis' 'Basketball_women' 'Bupa' 'Carcinogenesis'
#     'Chess' 'CiteSeer' 'ConsumerExpenditures' 'CORA' 'CraftBeer' 'Credit' 'cs' 'Dallas' 'DCG' 'Dunur'
#     'Elti' 'ErgastF1' 'Facebook' 'financial' 'ftp' 'geneea' 'genes' 'Hepatitis_std' 'Hockey' 'imdb_ijs'
#     'imdb_MovieLens' 'KRK' 'legalActs' 'medical' 'Mondial' 'Mooney_Family' 'MuskSmall' 'mutagenesis'
#     'nations' 'NBA' 'NCAA' 'Pima' 'PremierLeague' 'PTE' 'PubMed_Diabetes' 'Same_gen' 'SAP' 'SAT'
#     'Shakespeare' 'Student_loan' 'Toxicology' 'tpcc' 'tpcd' 'tpcds' 'trains' 'university' 'UTube'
#     'UW_std' 'VisualGenome' 'voc' 'WebKP' 'world')
# datasets=('Dallas' 'world')

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

NAME=$2
if [[ -z "$NAME" ]]; then
    NAME="benchmark"
fi

id="${NAME}_$(date '+%d-%m-%Y_%H:%M:%S')_$(openssl rand -hex 4)"

for d in ${try_datasets[@]}; do
    sbatch -o logs/$id/$d/batch.log $1 $id $d &
done

wait

