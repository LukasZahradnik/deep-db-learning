from db_transformer.data.dataset_defaults.fit_dataset_defaults import FIT_DATASET_DEFAULTS
from db_transformer.data.fit_dataset import FITRelationalDataset
from db_transformer.data.strategy.bfs import BFSStrategy
from db_transformer.data.utils import HeteroDataBuilder
from db_transformer.db.schema_autodetect import SchemaAnalyzer

dataset_name = 'CORA'

with FITRelationalDataset.create_remote_connection(dataset_name) as conn:
    schema_analyzer = SchemaAnalyzer(conn)

    schema = schema_analyzer.guess_schema()

    dataset = FITRelationalDataset(dataset_name, 'dataset', BFSStrategy(max_depth=4), schema=schema)

    defaults = FIT_DATASET_DEFAULTS[dataset_name]

    data = HeteroDataBuilder(conn,
                             schema,
                             target_table=defaults.target_table,
                             target_column=defaults.target_column,
                             target_one_hot=True).build()
    print(data)
