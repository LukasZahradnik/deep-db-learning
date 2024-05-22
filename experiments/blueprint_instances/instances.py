from typing import List, Dict, Any, Literal

from torch_geometric.typing import NodeType, EdgeType
from torch_geometric.data import HeteroData

from db_transformer.data import CTUDatasetDefault
from db_transformer.nn import BlueprintModel

from torch_frame import stype
from torch_frame.data import StatType

from .excelformer import create_excelformer_model
from .honza import create_honza_model
from .mlp import create_mlp_model
from .saint import create_saint_model
from .tabnet import create_tabnet_model
from .tabtransformer import create_tabtransformer_model
from .transformer import create_transformer_model
from .trompt import create_trompt_model


def create_blueprint_model(
    instance: Literal[
        "excelformer",
        "honza",
        "mlp",
        "saint",
        "tabnet",
        "tabtransformer",
        "transformer",
        "trompt",
    ],
    defaults: CTUDatasetDefault,
    col_names_dict: Dict[NodeType, Dict[stype, List[str]]],
    edge_types: List[EdgeType],
    col_stats_dict: Dict[NodeType, Dict[str, Dict[StatType, Any]]],
    config: Dict[str, Any],
) -> BlueprintModel:
    if instance == "excelformer":
        return create_excelformer_model(
            defaults, col_names_dict, edge_types, col_stats_dict, config
        )
    if instance == "honza":
        return create_honza_model(
            defaults, col_names_dict, edge_types, col_stats_dict, config
        )
    if instance == "mlp":
        return create_mlp_model(
            defaults, col_names_dict, edge_types, col_stats_dict, config
        )
    if instance == "saint":
        return create_saint_model(
            defaults, col_names_dict, edge_types, col_stats_dict, config
        )
    if instance == "tabnet":
        return create_tabnet_model(
            defaults, col_names_dict, edge_types, col_stats_dict, config
        )
    if instance == "tabtransformer":
        return create_tabtransformer_model(
            defaults, col_names_dict, edge_types, col_stats_dict, config
        )
    if instance == "transformer":
        return create_transformer_model(
            defaults, col_names_dict, edge_types, col_stats_dict, config
        )
    if instance == "trompt":
        return create_trompt_model(
            defaults, col_names_dict, edge_types, col_stats_dict, config
        )

    raise TypeError(f"Unknown blueprint instance named '{instance}'")
