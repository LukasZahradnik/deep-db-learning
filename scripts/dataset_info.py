import os, sys

sys.path.append(os.getcwd())

from typing import List, Dict, Union, Optional, Any, Tuple, Literal

import pandas as pd

import torch

import torch_geometric
from torch_geometric.typing import EdgeType, NodeType

import torch_frame

from db_transformer.data.ctu_dataset import CTUDataset, CTU_REPOSITORY_DEFAULTS

df_data: Dict[str, List[Any]] = dict(
    dataset=[],
    n_relations=[],
    n_edge_types=[],
    n_target_tuples=[],
    n_target_attributes=[],
    avg_target_edges=[],
    total_n_tuples=[],
    total_n_edges=[],
    total_ratio_edges_tuples=[],
    task=[],
)
for dataset_name in CTU_REPOSITORY_DEFAULTS:
    try:
        dataset = CTUDataset(dataset_name, data_dir="./datasets", force_remake=False)

        data, col_stats_dict = dataset.build_hetero_data(
            force_rematerilize=False, no_text_emebedding=False
        )

        target = dataset.defaults.target

        tf_dict: Dict[NodeType, torch_frame.TensorFrame] = data.collect("tf")
        edge_dict: Dict[EdgeType, torch_geometric.EdgeIndex] = data.collect("edge_index")
        target_table: torch_frame.TensorFrame = data[target[0]].tf

        df_data["dataset"].append(dataset.name)
        df_data["n_relations"].append(len(tf_dict))
        df_data["n_edge_types"].append(len(edge_dict) // 2)
        df_data["n_target_tuples"].append(target_table.num_rows)
        df_data["n_target_attributes"].append(target_table.num_cols)
        df_data["avg_target_edges"].append(
            sum(e.shape[1] for t, e in edge_dict.items() if t[0] == target[0])
            / target_table.num_rows
        )
        df_data["total_n_tuples"].append(sum(tf.num_rows for tf in tf_dict.values()))
        df_data["total_n_edges"].append(sum(e.shape[1] for e in edge_dict.values()) // 2)
        df_data["total_ratio_edges_tuples"].append(
            df_data["total_n_edges"][-1] / df_data["total_n_tuples"][-1]
        )
        df_data["task"].append(dataset.defaults.task.to_type())
    except Exception as e:
        print(f"Error processing {dataset_name}: {e}")

df = pd.DataFrame(df_data)

print(df)

df.to_csv("./datasets/info.csv", index=False)
