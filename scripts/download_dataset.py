from argparse import ArgumentParser
import os, sys
from typing import get_args

sys.path.append(os.getcwd())

from db_transformer.data.ctu_dataset import CTUDataset, CTUDatasetName

parser = ArgumentParser()
parser.add_argument("--dataset", type=str, choices=get_args(CTUDatasetName))

args = parser.parse_args()

dataset = CTUDataset(args.dataset, data_dir="./datasets", force_remake=True)

data, col_stats_dict = dataset.build_hetero_data(force_rematerilize=True)
