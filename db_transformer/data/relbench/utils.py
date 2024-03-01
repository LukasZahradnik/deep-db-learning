from typing import Dict

from torch_frame import stype

from db_transformer.schema.schema import ColumnDef, Schema, TableSchema
from db_transformer.schema import columns

COLUMN_DEF_STYPE: Dict[ColumnDef, stype] = {
    columns.CategoricalColumnDef: stype.categorical,
    columns.DateColumnDef: stype.timestamp,
    columns.DateTimeColumnDef: stype.timestamp,
    columns.NumericColumnDef: stype.numerical,
    columns.DurationColumnDef: stype.numerical,
    columns.TextColumnDef: stype.text_embedded,
    columns.OmitColumnDef: None,
}


def merge_schema_infered_stype(
    table_schema: TableSchema, infered_stype: Dict[str, stype]
) -> Dict[str, stype]:
    merged: Dict[str, stype] = {}
    for col_name, col_def in table_schema.columns.items():
        _stype = COLUMN_DEF_STYPE[type(col_def)]
        if _stype is None:
            _stype = infered_stype[col_name]
        merged[col_name] = _stype
    return merged
