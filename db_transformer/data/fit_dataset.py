from typing import Optional, Tuple, Literal

from sqlalchemy.engine import Connection, Engine, create_engine, make_url
from sqlalchemy.engine.url import URL
from sqlalchemy.exc import ArgumentError
from torch_geometric.data.dataset import Union

from db_transformer.data.convertor.schema_convertor import SchemaConvertor
from db_transformer.data.dataset import DBDataset
from db_transformer.data.dataset_defaults.fit_dataset_defaults import (
    FIT_DATASET_DEFAULTS,
    TaskType,
)
from db_transformer.data.strategy.bfs import BFSStrategy
from db_transformer.data.strategy.strategy import BaseStrategy
from db_transformer.db.schema_autodetect import SchemaAnalyzer
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
        cache_in_memory: bool = False,
    ):
        connection_url = self.get_url(database)

        if target_table is None or target_column is None:
            try:
                target_table = FIT_DATASET_DEFAULTS[database].target_table
                target_column = FIT_DATASET_DEFAULTS[database].target_column
            except KeyError:
                raise KeyError(
                    f"Relational FIT database '{database}' is unknown. "
                    "Please specify target_table and target_column explicitly."
                )

        super().__init__(
            database=database,
            target_table=target_table,
            target_column=target_column,
            connection_url=connection_url,
            root=root,
            strategy=strategy,
            download=True,
            verbose=verbose,
            schema=schema,
            cache_in_memory=cache_in_memory,
        )

    @classmethod
    def get_url(cls, dataset: str) -> str:
        connector = "mariadb+mysqlconnector"
        port = 3306
        return f"{connector}://guest:ctu-relational@78.128.250.186:{port}/{dataset}"

    @classmethod
    def create_remote_connection(cls, dataset: str):
        """Create a new SQLAlchemy Connection instance to the remote database.

        Create a new SQLAlchemy Connection instance to the remote database.
        Don't forget to close the Connection after you are done using it!
        """
        return Connection(create_engine(cls.get_url(dataset)))

    @classmethod
    def create_schema_analyzer(
        cls, dataset: str, connection: Connection, verbose=False, **kwargs
    ) -> SchemaAnalyzer:
        defaults = FIT_DATASET_DEFAULTS[dataset]
        target_type = (
            "categorical" if defaults.task == TaskType.CLASSIFICATION else "numeric"
        )
        return SchemaAnalyzer(
            connection,
            target=defaults.target,
            target_type=target_type,
            verbose=verbose,
            db_distinct_counter=defaults.db_distinct_counter,
            force_collation=defaults.force_collation,
            post_guess_schema_hook=defaults.schema_fixer,
            **kwargs,
        )

    @classmethod
    def get_target(cls, dataset: str) -> Tuple[str, str]:
        return FIT_DATASET_DEFAULTS[dataset].target
