from typing import List, Dict, Any, Literal

from torch_geometric.typing import NodeType, EdgeType
from torch_geometric.data import HeteroData

from db_transformer.data import CTUDatasetDefault
from db_transformer.nn import BlueprintModel

from torch_frame.data import StatType

from .honza_model import create_honza_model


def create_blueprint_model(
    instance: Literal["honza"],
    defaults: CTUDatasetDefault,
    col_names_dict: Dict[NodeType, List[str]],
    edge_types: List[EdgeType],
    col_stats_dict: Dict[NodeType, Dict[str, Dict[StatType, Any]]],
    config: Dict[str, Any],
) -> BlueprintModel:

    if instance == "honza":
        return create_honza_model(
            defaults, col_names_dict, edge_types, col_stats_dict, config
        )

    raise TypeError(f"Unknown blueprint instance named '{instance}'")
