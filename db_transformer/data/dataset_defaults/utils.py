from enum import Enum
from db_transformer.schema.schema import ForeignKeyDef, Schema


class TaskType(Enum):
    CLASSIFICATION = 1
    REGRESSION = 2
    LINK_PREDICTION = 3

    def to_type(self) -> str:
        if self.name == "CLASSIFICATION":
            return "categorical"
        elif self.name == "REGRESSION":
            return "numeric"
        else:
            return "edge_type"


def fix_citeseer_schema(schema: Schema):
    schema.cites.foreign_keys += [
        ForeignKeyDef(["cited_paper_id"], "paper", ["paper_id"]),
        ForeignKeyDef(["citing_paper_id"], "paper", ["paper_id"]),
    ]
