from typing import Dict

import lovely_tensors as lt
import torch
from db_transformer.data.embedder import CatEmbedder
from db_transformer.data.embedder.columns.num_embedder import NumEmbedder
from db_transformer.data.embedder.embedders import TableEmbedder
from db_transformer.data.fit_dataset import FITRelationalDataset
from db_transformer.data.utils.heterodata_builder import HeteroDataBuilder
from db_transformer.schema.columns import CategoricalColumnDef, NumericColumnDef
from torch_geometric.data.data import NodeType

DATASET_NAME = 'CORA'
DIM = 64

lt.monkey_patch()

def _expand_with_zeros(x_dict: Dict[NodeType, torch.Tensor]) -> Dict[NodeType, torch.Tensor]:
    out: Dict[NodeType, torch.Tensor]= {}

    for name, x in x_dict.items():
        if x.shape[-2] == 0:
            out[name] = torch.zeros((*x.shape[:-2], 1, x.shape[-1]), dtype=x.dtype)
        else:
            out[name] = x

    return out

with FITRelationalDataset.create_remote_connection(DATASET_NAME) as conn:
    print("Guessing schema...")
    schema = FITRelationalDataset.create_schema_analyzer(
        DATASET_NAME, conn, verbose=True).guess_schema()

    print(schema)

    print()
    print("Building HeteroData...")
    target_table, target_column = FITRelationalDataset.get_target(DATASET_NAME)

    data_builder = HeteroDataBuilder(
        conn,
        schema,
        target_table,
        target_column,
        fillna_with=0
    )

    data, column_defs, colnames = data_builder.build(with_column_names=True)

x_dict = data.collect('x')

print(data)

for name, x in x_dict.items():
    print(name.rjust(10), x)

print()
print("Embedding...")

embedder = TableEmbedder(
    (CategoricalColumnDef, lambda: CatEmbedder(DIM)),
    (NumericColumnDef, lambda: NumEmbedder(DIM)),
    dim=DIM,
    column_defs=column_defs,
    column_names=colnames,
)

x_dict = embedder(x_dict)

for name, x in x_dict.items():
    print(name.rjust(10), x)

print()
print("Expanding tables without features with zeros... (must come after embedding!)")
x_dict = _expand_with_zeros(x_dict)

for name, x in x_dict.items():
    print(name.rjust(10), x)

