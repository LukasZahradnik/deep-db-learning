from typing import Optional
from sqlalchemy.engine.url import URL

from torch_geometric.data.dataset import Union
from db_transformer.data.convertor.schema_convertor import SchemaConvertor

from db_transformer.data.dataset import DBDataset
from db_transformer.data.dataset_defaults.fit_dataset_defaults import FIT_DATASET_DEFAULTS
from db_transformer.data.strategy.bfs import BFSStrategy
from db_transformer.data.strategy.strategy import BaseStrategy
from db_transformer.schema import Schema


class FITRelationalDataset(DBDataset):
    def __init__(
        self,
        database: str,
        root: str,
        strategy: BaseStrategy,
        target_table: Optional[str] = None,
        target_column: Optional[str] = None,
        schema: Optional[Schema] = None,
        verbose=True,
        connector: str = "mariadb+mariadbconnector",
        cache_in_memory: bool = False,
    ):
        connection_url = f"{connector}://guest:relational@relational.fit.cvut.cz:3306/{database}"

        if target_table is None:
            try:
                target_table = FIT_DATASET_DEFAULTS[database].target_table
                target_column = FIT_DATASET_DEFAULTS[database].target_column
            except KeyError:
                raise KeyError(f"Relational FIT database '{database}' is unknown. Please explicitly specify target_table.")

        super().__init__(database=database,
                         target_table=target_table,
                         target_column=target_column,
                         connection_url=connection_url,
                         root=root,
                         strategy=strategy,
                         download=True,
                         verbose=verbose,
                         schema=schema,
                         cache_in_memory=cache_in_memory)


if __name__ == "__main__":
    test_dataset = FITRelationalDataset("mutagenesis", ".", strategy=BFSStrategy(1))
    # test_dataset = FITRelationalDataset('mutagenesis', target_table=None, root='.', strategy=BFSStrategy(18), dim=32)
    print(test_dataset.schema)
    print(test_dataset.len())
    print(test_dataset.get(6))
