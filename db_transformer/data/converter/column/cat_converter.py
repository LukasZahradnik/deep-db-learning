from typing import Any, Callable, Dict, List, Sequence, Tuple

import pandas as pd
import unidecode

from db_transformer.data.converter.column.series_converter import SeriesConverter
from db_transformer.schema.columns import CategoricalColumnDef
from db_transformer.schema.schema import ColumnDef

__ALL__ = ['CategoricalConverter']


class CategoricalConverter(SeriesConverter[CategoricalColumnDef]):
    def __call__(self,
                 column_def: CategoricalColumnDef,
                 column: pd.Series) -> Tuple[Sequence[pd.Series], Sequence[ColumnDef]]:
        distinct_vals, mapping = self._guess_value_set(column_def.card, column)

        # give None index of 0
        if None in distinct_vals:
            distinct_vals = [v for v in distinct_vals if v is not None]
            distinct_vals.insert(0, None)

        value_map = {v: i for i, v in enumerate(distinct_vals)}

        out_column = mapping(column).map(value_map)

        return (out_column, ), (column_def, )

    def _guess_value_set(self, cardinality: int, column: pd.Series) -> Tuple[List[Any],
                                                                             Callable[[pd.Series], pd.Series]]:
        MAPPINGS: List[Callable[[pd.Series], pd.Series]] = [
            lambda s: s,  # as-is
            lambda s: s.str.lower(),  # case insensitive
            lambda s: s.map(lambda s: unidecode.unidecode(s)),  # stripped accents
            lambda s: s.map(lambda s: unidecode.unidecode(s).lower()),  # stripped accents and CI
        ]

        choices: Dict[int, Tuple[List[Any], Callable[[pd.Series], pd.Series]]] = {}

        for mapping in MAPPINGS:
            try:
                values = mapping(column).unique().tolist()
                choices[len(values)] = values, mapping
            except AttributeError:
                pass

        if cardinality in choices:
            return choices[cardinality]

        errormsg = [f" ->    {vals} (cardinality {card})" for card, (vals, _) in choices.items()]

        raise RuntimeError(f"Expected {cardinality} unique values, "
                           f"but only found the following possible value sets:\n"
                           + '\n'.join(errormsg))
