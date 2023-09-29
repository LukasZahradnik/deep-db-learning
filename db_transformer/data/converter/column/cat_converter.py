from typing import Any, List, Optional, Sequence, Tuple, Union

import pandas as pd

from db_transformer.data.converter.column.series_converter import SeriesConverter
from db_transformer.db.distinct_cnt_retrieval import (
    SIMPLE_STRING_SERIES_MAPPERS,
    SeriesMapper,
    SimpleStringSeriesMapper,
    get_string_mapper,
)
from db_transformer.schema.columns import CategoricalColumnDef
from db_transformer.schema.schema import ColumnDef

__ALL__ = ['CategoricalConverter']


class CategoricalConverter(SeriesConverter[CategoricalColumnDef]):
    def __init__(self,
                 mapper: Optional[Union[SimpleStringSeriesMapper, SeriesMapper]] = None,
                 ) -> None:
        super().__init__()
        self.mapper = get_string_mapper(mapper) if mapper is not None else None

    def __call__(self,
                 column_def: CategoricalColumnDef,
                 column: pd.Series,
                 ) -> Tuple[Sequence[pd.Series], Sequence[ColumnDef]]:
        distinct_vals, mapper = self._guess_value_set(column_def.card, column)

        # give None index of 0
        if None in distinct_vals:
            distinct_vals = [v for v in distinct_vals if v is not None]
            distinct_vals.insert(0, None)

        value_map = {v: i for i, v in enumerate(distinct_vals)}

        out_column = mapper(column).map(value_map)

        return (out_column, ), (column_def, )

    def _guess_value_set(self, cardinality: int, column: pd.Series) -> Tuple[List[Any], SeriesMapper]:
        failed_mappings: List[Tuple[str, int, Optional[Exception]]] = []

        if self.mapper is not None:
            mappers = {'user_provided': self.mapper}
        else:
            mappers = SIMPLE_STRING_SERIES_MAPPERS

        for mapping_name, mapper in mappers.items():
            try:
                values = mapper(column).unique().tolist()
                if len(values) == cardinality:
                    return values, mapper
                failed_mappings.append((mapping_name, len(values), None))
            except AttributeError as e:
                failed_mappings.append((mapping_name, -1, e))

        def _exception_to_str(e: Optional[Exception]) -> str:
            if e is None:
                return ''

            return f" (failed: {e})"

        errormsg = [
            f" ->    {mapping_name} (cardinality {card}){_exception_to_str(e)}" for mapping_name, card, e in failed_mappings]

        raise RuntimeError(f"Expected {cardinality} unique values, "
                           f"but the following operations on values provided the following cardinalities instead:\n"
                           + '\n'.join(errormsg))
