import argparse
from typing import Literal
from typing import get_args as t_get_args

import torch.optim

from db_transformer.data.fit_dataset import FITRelationalDataset
from db_transformer.data.strategy.bfs import BFSStrategy
from db_transformer.transformer import DBTransformer

DatasetName = Literal["mutagenesis", "financial", "stats", "imdb_ijs", "CORA",
                      "trains", "Hepatitis_std", "genes", "UW_std", "PTE", "Toxicology", "Carcinogenesis"]

parser = argparse.ArgumentParser()
parser.add_argument("dataset", choices=t_get_args(DatasetName))
parser.add_argument("-d", "--depth", type=int, default=None, required=False)

args = parser.parse_args()

dataset_name: DatasetName = args.dataset

if dataset_name == "mutagenesis":
    dataset = FITRelationalDataset(dataset_name, ".", strategy=BFSStrategy(args.depth if args.depth is not None else 18))
    print(dataset.schema)
    print(dataset.get(0))
    print(dataset.get(6))

# # NOTE: large database!
# #
elif dataset_name == "financial":
    dataset = FITRelationalDataset(dataset_name, ".", strategy=BFSStrategy(args.depth if args.depth is not None else 4))
    print(dataset.schema)
    print(dataset.get(6))

# # NOTE: large database!
# #
elif dataset_name == "stats":
    dataset = FITRelationalDataset(dataset_name, ".", strategy=BFSStrategy(args.depth if args.depth is not None else 4))
    print(dataset.schema)
    print(dataset.get(0))
    print(dataset.get(6))

# # NOTE: large database!
# #
elif dataset_name == "imdb_ijs":
    dataset = FITRelationalDataset(dataset_name, ".", strategy=BFSStrategy(args.depth if args.depth is not None else 4))
    print(dataset.schema)
    print(dataset.get(0))

elif dataset_name == "CORA":
    dataset = FITRelationalDataset(dataset_name, ".", strategy=BFSStrategy(args.depth if args.depth is not None else 4))
    print(dataset.schema)
    print(dataset.get(0))

elif dataset_name == "trains":
    dataset = FITRelationalDataset(dataset_name, ".", strategy=BFSStrategy(args.depth if args.depth is not None else 8))
    print(dataset.schema)
    print(dataset.get(4))

elif dataset_name == "Hepatitis_std":
    dataset = FITRelationalDataset(dataset_name, ".", strategy=BFSStrategy(args.depth if args.depth is not None else 4))
    print(dataset.schema)
    print(dataset.get(1))
    print(dataset.get(0))

elif dataset_name == "genes":
    dataset = FITRelationalDataset(dataset_name, ".", strategy=BFSStrategy(args.depth if args.depth is not None else 12))
    print(dataset.schema)
    print(dataset.get(0))

elif dataset_name == "UW_std":
    dataset = FITRelationalDataset(dataset_name, ".", strategy=BFSStrategy(args.depth if args.depth is not None else 4))
    print(dataset.schema)
    print(dataset.get(2))
    print(dataset.get(0))

elif dataset_name == "PTE":
    dataset = FITRelationalDataset(dataset_name, ".", strategy=BFSStrategy(args.depth if args.depth is not None else 4))
    print(dataset.schema)
    print(dataset.get(7))

elif dataset_name == "Toxicology":
    dataset = FITRelationalDataset(dataset_name, ".", strategy=BFSStrategy(args.depth if args.depth is not None else 9))
    print(dataset.schema)
    print(dataset.get(0))

elif dataset_name == "Carcinogenesis":
    dataset = FITRelationalDataset(dataset_name, ".", strategy=BFSStrategy(args.depth if args.depth is not None else 4))
    print(dataset.schema)
    print(dataset.get(5))
