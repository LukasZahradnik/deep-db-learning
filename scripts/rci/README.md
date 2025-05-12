## GNN
```bash
sbatch -o logs/experiment_gnn_$(date '+%d-%m-%Y_%H:%M:%S').log scripts/rci/experiment_gnn.sh
```

## TabNet
```bash
sbatch -o logs/experiment_tabnet_$(date '+%d-%m-%Y_%H:%M:%S').log scripts/rci/experiment_tabnet.sh
```

## DBFormer
```bash
sbatch -o logs/experiment_dbformer_$(date '+%d-%m-%Y_%H:%M:%S').log scripts/rci/experiment_dbformer.sh
sbatch -o logs/experiment_dbformer_basic_$(date '+%d-%m-%Y_%H:%M:%S').log scripts/rci/experiment_dbformer_basic.sh
sbatch -o logs/experiment_dbformer_text_$(date '+%d-%m-%Y_%H:%M:%S').log scripts/rci/experiment_dbformer_text.sh
sbatch -o logs/experiment_dbformer_time_$(date '+%d-%m-%Y_%H:%M:%S').log scripts/rci/experiment_dbformer_time.sh
```

## TabTransfomer
```bash
sbatch -o logs/experiment_tabtransformer_$(date '+%d-%m-%Y_%H:%M:%S').log scripts/rci/experiment_tabtransformer.sh
```

## SAINT
```bash
sbatch -o logs/experiment_saint_$(date '+%d-%m-%Y_%H:%M:%S').log scripts/rci/experiment_saint.sh
```

## Trompt
```bash
sbatch -o logs/experiment_trompt_$(date '+%d-%m-%Y_%H:%M:%S').log scripts/rci/experiment_trompt.sh
```

## MLP
```bash
sbatch -o logs/experiment_mlp_$(date '+%d-%m-%Y_%H:%M:%S').log scripts/rci/experiment_mlp.sh
```


## getml XGBoost
```bash
sbatch -w n05 -o logs/experiment_cilp_$(date '+%d-%m-%Y_%H:%M:%S').log scripts/rci/experiment_getml.sh 1 5
sbatch -w n06 -o logs/experiment_getml_$(date '+%d-%m-%Y_%H:%M:%S').log scripts/rci/experiment_getml.sh 6 10
sbatch -w n07 -o logs/experiment_getml_$(date '+%d-%m-%Y_%H:%M:%S').log scripts/rci/experiment_getml.sh 11 15
sbatch -w n08 -o logs/experiment_getml_$(date '+%d-%m-%Y_%H:%M:%S').log scripts/rci/experiment_getml.sh 15 20
sbatch -w n11 -o logs/experiment_getml_$(date '+%d-%m-%Y_%H:%M:%S').log scripts/rci/experiment_getml.sh 21 26
```


## CILP++
```bash
sbatch -w n11 -o logs/experiment_cilp_$(date '+%d-%m-%Y_%H:%M:%S').log scripts/rci/experiment_cilp.sh 1 5
sbatch -w n02 -o logs/experiment_cilp_$(date '+%d-%m-%Y_%H:%M:%S').log scripts/rci/experiment_cilp.sh 6 10
sbatch -w n10 -o logs/experiment_cilp_$(date '+%d-%m-%Y_%H:%M:%S').log scripts/rci/experiment_cilp.sh 11 15
sbatch -w n17 -o logs/experiment_cilp_$(date '+%d-%m-%Y_%H:%M:%S').log scripts/rci/experiment_cilp.sh 15 20
sbatch -w n18 -o logs/experiment_cilp_$(date '+%d-%m-%Y_%H:%M:%S').log scripts/rci/experiment_cilp.sh 21 26
```

## SRLBoost
```bash
sbatch -o logs/experiment_srlboost_$(date '+%d-%m-%Y_%H:%M:%S').log scripts/rci/experiment_srlboost.sh
```