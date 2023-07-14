from typing import Optional

from db_transformer.ndata.dataset import DBDataset
from db_transformer.ndata.dataset_defaults.fit_dataset_defaults import FIT_DATASET_DEFAULTS
from db_transformer.ndata.strategy.bfs import BFSStrategy
from db_transformer.ndata.strategy.strategy import BaseStrategy
from db_transformer.schema import Schema


class FITRelationalDataset(DBDataset):
    def __init__(
        self, database: str, target_table: Optional[str], root: str, strategy: BaseStrategy, schema: Optional[Schema] = None,
        connector: str = "mariadb+mariadbconnector",
    ):
        connection_url = f"{connector}://guest:relational@relational.fit.cvut.cz:3306/{database}"

        if target_table is None:
            try:
                target_table = FIT_DATASET_DEFAULTS[database].target_table
            except KeyError:
                raise KeyError(f"Relational FIT database '{database}' is unknown. Please explicitly specify target_table.")

        super().__init__(database=database,
                         target_table=target_table,
                         connection_url=connection_url,
                         root=root,
                         strategy=strategy,
                         download=True,
                         schema=schema)


if __name__ == "__main__":
    test_dataset = FITRelationalDataset('mutagenesis', target_table=None, root='.', strategy=BFSStrategy(18))
    print(test_dataset.schema)
    print(test_dataset.len())
    print(test_dataset.get(6))
