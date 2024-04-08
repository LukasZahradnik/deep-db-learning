from typing import List, Dict, Any, Literal

from torch_geometric.data import HeteroData

from db_transformer.data import CTUDatasetDefault
from db_transformer.nn import BlueprintModel

from .honza_model import create_honza_model


def create_blueprint_model(
    instance: Literal["honza"],
    defaults: CTUDatasetDefault,
    data: HeteroData,
    config: Dict[str, Any],
) -> BlueprintModel:

    if instance == "honza":
        return create_honza_model(defaults, data, config)

    raise TypeError(f"Unknown blueprint instance named '{instance}'")
