import argparse
import os
import random
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Callable, Literal, Optional, TypeVar
from typing import get_args as t_get_args

import getml
import lovely_tensors as lt
import mlflow
import numpy as np
import pandas as pd
import torch
import torch_geometric.transforms as T
from db_transformer.data.dataset_defaults.fit_dataset_defaults import FIT_DATASET_DEFAULTS, TaskType
from db_transformer.data.fit_dataset import FITRelationalDataset
from db_transformer.data.utils import HeteroDataBuilder
from db_transformer.schema.columns import (
    CategoricalColumnDef,
    DateColumnDef,
    DateTimeColumnDef,
    DurationColumnDef,
    NumericColumnDef,
    OmitColumnDef,
    TextColumnDef,
    TimeColumnDef,
)
from db_transformer.schema.schema import ForeignKeyDef, Schema
from getml.feature_learning import loss_functions
from simple_parsing import ArgumentParser, DashVariant
from sqlalchemy.engine import Connection

lt.monkey_patch()

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


DatasetType = Literal[
    'Accidents', 'Airline', 'Atherosclerosis', 'Basketball_women', 'Bupa', 'Carcinogenesis',
    'Chess', 'CiteSeer', 'ConsumerExpenditures', 'CORA', 'CraftBeer', 'Credit', 'cs', 'Dallas', 'DCG', 'Dunur',
    'Elti', 'ErgastF1', 'Facebook', 'financial', 'ftp', 'geneea', 'genes', 'Hepatitis_std', 'Hockey', 'imdb_ijs',
    'imdb_MovieLens', 'KRK', 'legalActs', 'medical', 'Mondial', 'Mooney_Family', 'MuskSmall', 'mutagenesis',
    'nations', 'NBA', 'NCAA', 'Pima', 'PremierLeague', 'PTE', 'PubMed_Diabetes', 'Same_gen', 'SAP', 'SAT',
    'Shakespeare', 'Student_loan', 'Toxicology', 'tpcc', 'tpcd', 'tpcds', 'trains', 'university', 'UTube',
    'UW_std', 'VisualGenome', 'voc', 'WebKP', 'world'
]


DEFAULT_DATASET_NAME: DatasetType = 'CORA'


@dataclass
class DataConfig:
    pass


@dataclass
class ModelConfig:
    depths: list[int]


_T = TypeVar('_T')


def build_data(
        c: Callable[[HeteroDataBuilder], _T],
        dataset=DEFAULT_DATASET_NAME,
        conn: Optional[Connection] = None,
        schema: Optional[Schema] = None,
        device=None) -> _T:
    has_sub_connection = conn is None

    if has_sub_connection:
        conn = FITRelationalDataset.create_remote_connection(dataset)

    defaults = FIT_DATASET_DEFAULTS[dataset]

    if schema is None:
        schema = FITRelationalDataset.create_schema_analyzer(
            dataset, conn, verbose=True).guess_schema()

    builder = HeteroDataBuilder(conn,
                                schema,
                                target_table=defaults.target_table,
                                target_column=defaults.target_column,
                                separate_target=True,
                                create_reverse_edges=True,
                                fillna_with=0.0,
                                device=device)
    out = c(builder)

    if has_sub_connection:
        conn.close()

    return out


def create_data(dataset=DEFAULT_DATASET_NAME, data_config: Optional[DataConfig] = None, device=None):
    if data_config is None:
        data_config = DataConfig()

    with FITRelationalDataset.create_remote_connection(dataset) as conn:
        defaults = FIT_DATASET_DEFAULTS[dataset]

        schema_analyzer = FITRelationalDataset.create_schema_analyzer(dataset, conn, verbose=True)
        schema = schema_analyzer.guess_schema()

        data_pd, (data, column_defs, colnames) = build_data(
            lambda builder: (builder._get_dataframes_raw(), builder.build(with_column_names=True)),
            dataset=dataset,
            conn=conn,
            schema=schema,
            # device=device
        )

        n_total = data[defaults.target_table].x.shape[0]
        T.RandomNodeSplit('train_rest', num_val=int(0.30 * n_total), num_test=0)(data)

        return data, data_pd, schema, defaults, column_defs, colnames


TARGET_TABLE = '__target_table'


def label_getml_roles(data_pd: dict[str, pd.DataFrame], schema: Schema, target_table: str, target_column: str, task: TaskType):
    data: dict[str, getml.data.DataFrame] = {}

    # for name, df in data_pd.items():
    #     data[name] = getml.data.DataFrame.from_pandas(df, name=name)

    conn = getml.database.connect_mariadb(
        host="relational.fit.cvut.cz",
        dbname="CORA",
        port=3306,
        user="guest",
        password="relational"
    )

    def load_if_needed(name):
        if not getml.data.exists(name):
            data_frame = getml.data.DataFrame.from_db(
                name=name,
                table_name=name,
                conn=conn
            )
            data_frame.save()
        else:
            data_frame = getml.data.load_data_frame(name)
        return data_frame

    for tname in data_pd:
        data[tname] = load_if_needed(tname)

    # add proper roles based on schema
    for table_name, table_def in schema.items():
        assert table_name in data
        table = data[table_name]

        # foreign key roles
        for fk_def in table_def.foreign_keys:
            table.set_role(fk_def.columns, getml.data.roles.join_key)
            data[fk_def.ref_table].set_role(fk_def.ref_columns, getml.data.roles.join_key)
            for c in fk_def.columns:
                table_def.columns[c] = OmitColumnDef()
            for c in fk_def.ref_columns:
                schema[fk_def.ref_table].columns[c] = OmitColumnDef()

        for column_name, column_def in table_def.columns.items():
            if table_name == target_table and column_name == target_column:
                continue

            if isinstance(column_def, CategoricalColumnDef):

                # # remap strings to integers because getml won't do it for us in community edition or whatever
                # if table[column_name] == getml.data.roles.unused_string:
                #     col_vals_unique = table[column_name].unique()
                #     col_vals_unique.sort()
                #     col_idx_map = {v: i for i, v in enumerate(col_vals_unique)}
                #     table[column_name] = pd.Series(table[column_name]).map(col_idx_map).to_numpy()

                role = getml.data.roles.categorical
            elif isinstance(column_def, (
                    NumericColumnDef, DateColumnDef, DateTimeColumnDef, DurationColumnDef, TimeColumnDef)):
                role = getml.data.roles.numerical
            elif isinstance(column_def, TextColumnDef):
                role = getml.data.roles.text
            else:
                role = None

            if role is not None:
                print(column_name, role)
                table.set_role(column_name, role)

    if task == TaskType.CLASSIFICATION and data_pd[target_table][target_column].nunique() > 2:
        data[TARGET_TABLE] = getml.data.make_target_columns(data[target_table].copy(name=TARGET_TABLE), target_column)
    else:
        data[TARGET_TABLE] = data[target_table].copy(name=TARGET_TABLE)
        data[TARGET_TABLE].set_role(target_column, getml.data.roles.target)

    for name, df in data.items():
        for col in df.columns:
            print(name, col, type(df[col]))

    return data


GetmlTableIdentifier = tuple[str, int | None]


@dataclass(frozen=True)
class OrderedFKDef:
    unique_id: int
    table: str
    ref_table: str
    columns: list[str]
    ref_columns: list[str]
    relationship: str = getml.data.relationship.many_to_one

    def __hash__(self) -> int:
        return self.unique_id

    @classmethod
    def create_from(cls, table: str, fk_def: ForeignKeyDef, unique_id: int) -> 'OrderedFKDef':
        return OrderedFKDef(
            unique_id=unique_id,
            table=table,
            ref_table=fk_def.ref_table,
            columns=fk_def.columns,
            ref_columns=fk_def.ref_columns,
        )

    def invert(self, unique_id: int) -> 'OrderedFKDef':
        return OrderedFKDef(
            unique_id=unique_id,
            table=self.ref_table,
            ref_table=self.table,
            columns=self.ref_columns,
            ref_columns=self.columns,
            relationship=(getml.data.relationship.one_to_many if (
                self.relationship == getml.data.relationship.many_to_one) else getml.data.relationship.many_to_one)
        )


@dataclass
class GetmlFKDef:
    table: GetmlTableIdentifier
    ref_table: GetmlTableIdentifier
    columns: list[str]
    ref_columns: list[str]
    relationship: str = getml.data.relationship.many_to_one

    def invert(self) -> 'GetmlFKDef':
        return GetmlFKDef(
            table=self.ref_table,
            ref_table=self.table,
            columns=self.ref_columns,
            ref_columns=self.columns,
            relationship=(getml.data.relationship.one_to_many if (
                self.relationship == getml.data.relationship.many_to_one) else getml.data.relationship.many_to_one)
        )

    def __repr__(self) -> str:
        return f"FK({self.table} -> {self.ref_table})"


def bfs(schema: Schema, target_table: str, max_depth: int) -> tuple[set[GetmlTableIdentifier], list[GetmlFKDef]]:
    # init
    fk_defs: dict[str, list[OrderedFKDef]] = defaultdict(lambda: [])
    _i = 0
    for table_name, table_def in schema.items():
        for fk_def in table_def.foreign_keys:
            ofk_def = OrderedFKDef.create_from(table_name, fk_def, unique_id=_i)
            fk_defs[table_name] += [ofk_def]
            fk_defs[fk_def.ref_table] += [ofk_def.invert(unique_id=_i)]
            _i += 1

    out_nodes: set[GetmlTableIdentifier] = set()
    out_edges: list[GetmlFKDef] = []

    def _next_identifier(n: str, max_idxs: dict[str, int], max_idxs_max: dict[str, int]) -> GetmlTableIdentifier:
        i = max_idxs[n] + 1
        max_idxs[n] += 1
        if max_idxs[n] > max_idxs_max[n]:
            max_idxs_max[n] = max_idxs[n]
        return n, i

    # bfs itself

    _max_idxs: dict[str, int] = {k: -1 for k in schema}

    _nodes_this_layer: set[GetmlTableIdentifier] = {_next_identifier(target_table, _max_idxs, _max_idxs)}
    out_nodes.update(_nodes_this_layer)

    for _ in range(max_depth):
        if len(_nodes_this_layer) == 0:
            break

        nodes_next_layer: set[GetmlTableIdentifier] = set()
        max_idxs_next: dict[str, int] = _max_idxs.copy()

        # build edges and nodes for next layer
        for n, i in _nodes_this_layer:
            max_idxs_this = _max_idxs.copy()
            for ofk_def in fk_defs[n]:
                id2 = _next_identifier(ofk_def.ref_table, max_idxs_this, max_idxs_next)
                nodes_next_layer.add(id2)
                out_edges += [
                    GetmlFKDef(
                        table=(n, i),
                        ref_table=id2,
                        columns=ofk_def.columns,
                        ref_columns=ofk_def.ref_columns,
                        relationship=ofk_def.relationship,
                    )
                ]

        # refresh nodes set for next layer
        _nodes_this_layer = nodes_next_layer
        _max_idxs = max_idxs_next
        out_nodes.update(nodes_next_layer)

    # after-bfs fixes

    # rename nodes that do not repeat to (n, None) from (n, 0):

    # build the data
    zero_only_nodes = {n for (n, _) in out_nodes}
    for n, i in out_nodes:
        if i is not None and i > 0:
            zero_only_nodes.discard(n)

    # rename in nodes
    for n in zero_only_nodes:
        out_nodes.remove((n, 0))
        out_nodes.add((n, None))

    # rename in edges
    for e in out_edges:
        if e.table[0] in zero_only_nodes:
            e.table = e.table[0], None
        if e.ref_table[0] in zero_only_nodes:
            e.ref_table = e.ref_table[0], None

    # rename target table in nodes
    out_nodes_old = out_nodes
    out_nodes = set()

    def _target_table_new_name(n: str, i: int | None) -> GetmlTableIdentifier:
        if n != target_table:
            return n, i
        elif i is None or i == 0:
            return TARGET_TABLE, None
        else:
            return n, i - 1

    for n, i in out_nodes_old:
        out_nodes.add(_target_table_new_name(n, i))

    # rename target table in edges
    for e in out_edges:
        e.table = _target_table_new_name(*e.table)
        e.ref_table = _target_table_new_name(*e.ref_table)

    return out_nodes, out_edges


def build_getml_datamodel(data_pd: dict[str, getml.data.DataFrame],
                          nodes: set[GetmlTableIdentifier],
                          edges: list[GetmlFKDef]) -> getml.data.DataModel:
    dm = getml.data.DataModel(data_pd[TARGET_TABLE].to_placeholder(TARGET_TABLE))

    dfs_repeated: dict[str, getml.data.DataFrame | list[getml.data.DataFrame]] = {}

    for n, i in nodes:
        if n == TARGET_TABLE:
            continue

        if i is None:
            dfs_repeated[n] = data_pd[n]
        elif n not in dfs_repeated:
            dfs_repeated[n] = [data_pd[n]]
        elif isinstance(dfs_repeated[n], list):
            dfs_repeated[n] += [data_pd[n]]
        else:
            raise RuntimeError()

    dm.add(getml.data.to_placeholder(**dfs_repeated))

    for fk in edges:
        p_left: getml.data.Placeholder = getattr(dm, fk.table[0])
        if fk.table[1] is not None:
            p_left = p_left[fk.table[1]]

        p_right: getml.data.Placeholder = getattr(dm, fk.ref_table[0])
        if fk.ref_table[1] is not None:
            p_right = p_right[fk.ref_table[1]]

        p_left.join(
            p_right,
            on=list(zip(fk.columns, fk.ref_columns)),
            relationship=fk.relationship,
        )

    return dm


def strip_targets_prefix(targets: list[str], target_column: str) -> list[str]:
    out: list[str] = []

    prefix = f"{target_column}="

    for t in targets:
        if t.startswith(prefix):
            t = t[len(prefix):]
        out.append(t)
    return out


def evaluate_accuracy(targets: list[str], target_column: str, y_pred_prob: np.ndarray, y_true: pd.Series) -> float:
    targets = strip_targets_prefix(targets, target_column)
    targets_idx_map = {v: i for i, v in enumerate(targets)}
    y_pred = np.argmax(y_pred_prob, 1)
    y_true_np = y_true.map(targets_idx_map).to_numpy()
    print(list(y_pred), y_pred.shape)
    print(list(y_true_np), y_true_np.shape)
    return np.mean(y_pred == y_true_np)


def main(dataset_name: str, data_config: DataConfig, model_config: ModelConfig):
    data, data_pd, schema, defaults, column_defs, colnames = create_data(dataset_name, data_config)

    assert defaults.task == TaskType.CLASSIFICATION

    data_pd = label_getml_roles(data_pd, schema, defaults.target_table, defaults.target_column, defaults.task)

    for name, tbl in data_pd.items():
        print(name, tbl)

    # build split
    # split = getml.data.split.random(train=0.7, test=0.3)  # TODO use universal split
    split = pd.Series(data[defaults.target_table].train_mask.numpy())
    split = split.map({True: 'train', False: 'test'}).to_frame('split')
    split = getml.data.DataFrame.from_pandas(split, 'split')['split']

    container = getml.data.Container(population=data_pd[TARGET_TABLE], split=split)
    container.add(**{k: v for k, v in data_pd.items() if k != TARGET_TABLE})
    container.freeze()

    for max_depth in model_config.depths:
        nodes, edges = bfs(schema, defaults.target_table, max_depth=max_depth)
        print(nodes)
        print(edges)
        dm = build_getml_datamodel(data_pd, nodes, edges)
        print(dm)

        mapping = getml.preprocessors.Mapping()

        fast_prop = getml.feature_learning.FastProp(
            loss_function=loss_functions.CrossEntropyLoss,
        )

        feature_selector = getml.predictors.XGBoostClassifier()
        predictor = getml.predictors.XGBoostClassifier()

        pipe = getml.pipeline.Pipeline(
            data_model=dm,
            preprocessors=[mapping],
            feature_learners=[fast_prop],
            # feature_selectors=[feature_selector],
            predictors=[predictor],
            share_selected_features=0.5,
        )

        pipe.fit(container.train)
        print(pipe.score(container.train))
        y_pred_prob = pipe.predict(container.train)
        assert y_pred_prob is not None
        print("train_acc:", evaluate_accuracy(
            pipe.targets,
            defaults.target_column,
            y_pred_prob,
            data_pd[defaults.target_table][split == 'train'].to_pandas()[defaults.target_column],
        ))
        print(pipe.score(container.test))
        y_pred_prob = pipe.predict(container.test)
        print("test_acc:", evaluate_accuracy(
            pipe.targets,
            defaults.target_column,
            y_pred_prob,
            data_pd[defaults.target_table][split == 'test'].to_pandas()[defaults.target_column],
        ))

    return split, container, data_pd, pipe  # TODO remove


if __name__ == "__main__":
    parser = ArgumentParser(add_option_string_dash_variants=DashVariant.DASH)
    parser.add_argument('dataset', choices=t_get_args(DatasetType))
    parser.add_arguments(ModelConfig, dest="model_config")
    parser.add_arguments(DataConfig, dest="data_config")
    parser.add_argument("--mlflow", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args([DEFAULT_DATASET_NAME, "--depths", "4"])
    model_config: ModelConfig = args.model_config
    data_config: DataConfig = args.data_config
    dataset_name: DatasetType = args.dataset
    do_mlflow: bool = args.mlflow

    def _run_main():
        main(dataset_name, data_config, model_config)

    dataset_idx = sorted(FIT_DATASET_DEFAULTS).index(dataset_name)
    getml.communication.port.value = 11111 + dataset_idx
    getml.communication.tcp_port.value = 12111 + dataset_idx

    getml.engine.launch(launch_browser=True,
                        project_directory='/mnt/personal/neumaja5/getml-internal-data/',
                        http_port=getml.communication.port.value,
                        tcp_port=getml.communication.tcp_port.value)
    getml.engine.set_project(dataset_name)

    if do_mlflow:
        os.environ['MLFLOW_TRACKING_URI'] = 'http://147.32.83.171:2222'
        mlflow.set_experiment("deep_rl_learning NEW - neumaja5")
        mlflow.pytorch.autolog()

        file_name = os.path.basename(__file__)
        with mlflow.start_run(run_name=f"{file_name} - {dataset_name} - {uuid.uuid4()}") as run:
            mlflow.set_tag('dataset', dataset_name)
            mlflow.set_tag('Model Source', file_name)

            for k, v in asdict(model_config).items():
                mlflow.log_param(k, v)

            for k, v in asdict(data_config).items():
                mlflow.log_param(k, v)

            try:
                _run_main()
            except Exception as ex:
                mlflow.set_tag('exception', str(ex))
                raise ex
            finally:
                getml.engine.shutdown()
    else:
        try:
            _run_main()
        finally:
            getml.engine.shutdown()
