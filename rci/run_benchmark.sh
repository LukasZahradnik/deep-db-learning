#!/bin/bash

datasets=('Accidents' 'Airline' 'Atherosclerosis' 'Basketball_women' 'Bupa' 'Carcinogenesis'
    'Chess' 'CiteSeer' 'ConsumerExpenditures' 'CORA' 'CraftBeer' 'Credit' 'cs' 'Dallas' 'DCG' 'Dunur'
    'Elti' 'ErgastF1' 'Facebook' 'financial' 'ftp' 'geneea' 'genes' 'Hepatitis_std' 'Hockey' 'imdb_ijs'
    'imdb_MovieLens' 'KRK' 'legalActs' 'medical' 'Mondial' 'Mooney_Family' 'MuskSmall' 'mutagenesis'
    'nations' 'NBA' 'NCAA' 'Pima' 'PremierLeague' 'PTE' 'PubMed_Diabetes' 'Same_gen' 'SAP' 'SAT'
    'Shakespeare' 'Student_loan' 'Toxicology' 'tpcc' 'tpcd' 'tpcds' 'trains' 'university' 'UTube'
    'UW_std' 'VisualGenome' 'voc' 'WebKP' 'world')
# datasets=('Dallas' 'world')

id="simple_$(openssl rand -hex 4)_$(date '+%d-%m-%Y_%H:%M:%S')"

for d in ${datasets[@]}; do
    sbatch -o logs/$id/$d/batch.log rci/dataset.batch $id $d &
done

wait

