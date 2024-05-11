import argparse
from datetime import datetime
from pathlib import Path
import traceback
import os, sys

sys.path.append(os.getcwd())
import random
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Callable, Literal, Optional, TypeVar, get_args
from typing import get_args as t_get_args

import getml
import mlflow
from mlflow.entities import Param
from mlflow.utils.mlflow_tags import MLFLOW_USER, MLFLOW_PARENT_RUN_ID

import numpy as np
import pandas as pd
import torch
import torch_geometric.transforms as T
from db_transformer.data import (
    CTUDataset,
    CTUDatasetName,
    CTU_REPOSITORY_DEFAULTS,
    CTUDatasetDefault,
    TaskType,
)
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
from sqlalchemy.engine import Connection

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


DEFAULT_DATASET_NAME: CTUDatasetName = "CORA"

DEFAULT_EXPERIMENT_NAME = "pelesjak-deep-db-tests"

RANDOM_SEED = 42

TARGET_TABLE = "__target_table"


def label_getml_roles(
    data_pd: dict[str, pd.DataFrame],
    schema: Schema,
    defaults: CTUDatasetDefault,
):
    df_dict: dict[str, getml.data.DataFrame] = {}

    for tname, df in data_pd.items():
        df_dict[tname] = getml.data.DataFrame.from_pandas(df, name=tname)

    # add proper roles based on schema
    for table_name, table_def in schema.items():
        assert table_name in df_dict
        table = df_dict[table_name]

        # foreign key roles
        for fk_def in table_def.foreign_keys:
            table.set_role(fk_def.columns, getml.data.roles.join_key)
            df_dict[fk_def.ref_table].set_role(
                fk_def.ref_columns, getml.data.roles.join_key
            )
            for c in fk_def.columns:
                table_def.columns[c] = OmitColumnDef()
            for c in fk_def.ref_columns:
                schema[fk_def.ref_table].columns[c] = OmitColumnDef()

        if "__filler" in table.colnames:
            table.set_role("__filler", getml.data.roles.categorical)

        for column_name, column_def in table_def.columns.items():
            if (
                table_name == defaults.target_table
                and column_name == defaults.target_column
            ):
                continue

            if isinstance(column_def, CategoricalColumnDef):

                # # remap strings to integers because getml won't do it for us in community edition or whatever
                # if table[column_name] == getml.data.roles.unused_string:
                #     col_vals_unique = table[column_name].unique()
                #     col_vals_unique.sort()
                #     col_idx_map = {v: i for i, v in enumerate(col_vals_unique)}
                #     table[column_name] = pd.Series(table[column_name]).map(col_idx_map).to_numpy()

                role = getml.data.roles.categorical
            elif isinstance(
                column_def,
                (
                    NumericColumnDef,
                    DateColumnDef,
                    DateTimeColumnDef,
                    DurationColumnDef,
                    TimeColumnDef,
                ),
            ):
                role = getml.data.roles.numerical
            elif isinstance(column_def, TextColumnDef):
                role = getml.data.roles.text
            else:
                role = None

            if role is not None:
                print(column_name, role)
                table.set_role(column_name, role)

    if (
        defaults.task
        == TaskType.CLASSIFICATION
        # and data_pd[defaults.target_table][defaults.target_column].nunique() > 2
    ):
        base = df_dict[defaults.target_table].copy(name=TARGET_TABLE)
        unique_values = base[defaults.target_column].unique()
        df_dict[TARGET_TABLE] = base

        for label in unique_values:
            col = (base[defaults.target_column] == label).as_num()
            name = defaults.target_column + "=" + str(label)
            df_dict[TARGET_TABLE] = df_dict[TARGET_TABLE].with_column(
                col=col, name=name, role=getml.data.roles.target
            )
    else:
        df_dict[TARGET_TABLE] = df_dict[defaults.target_table].copy(name=TARGET_TABLE)
        df_dict[TARGET_TABLE].set_role(defaults.target_column, getml.data.roles.target)

    for name, df in df_dict.items():
        for col in df.columns:
            print(name, col, type(df[col]))

    return df_dict


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
    def create_from(
        cls, table: str, fk_def: ForeignKeyDef, unique_id: int
    ) -> "OrderedFKDef":
        return OrderedFKDef(
            unique_id=unique_id,
            table=table,
            ref_table=fk_def.ref_table,
            columns=fk_def.columns,
            ref_columns=fk_def.ref_columns,
        )

    def invert(self, unique_id: int) -> "OrderedFKDef":
        return OrderedFKDef(
            unique_id=unique_id,
            table=self.ref_table,
            ref_table=self.table,
            columns=self.ref_columns,
            ref_columns=self.columns,
            relationship=(
                getml.data.relationship.one_to_many
                if (self.relationship == getml.data.relationship.many_to_one)
                else getml.data.relationship.many_to_one
            ),
        )


@dataclass
class GetmlFKDef:
    table: GetmlTableIdentifier
    ref_table: GetmlTableIdentifier
    columns: list[str]
    ref_columns: list[str]
    relationship: str = getml.data.relationship.many_to_one

    def invert(self) -> "GetmlFKDef":
        return GetmlFKDef(
            table=self.ref_table,
            ref_table=self.table,
            columns=self.ref_columns,
            ref_columns=self.columns,
            relationship=(
                getml.data.relationship.one_to_many
                if (self.relationship == getml.data.relationship.many_to_one)
                else getml.data.relationship.many_to_one
            ),
        )

    def __repr__(self) -> str:
        return f"FK({self.table} -> {self.ref_table})"


def bfs(
    schema: Schema, target_table: str, max_depth: int
) -> tuple[set[GetmlTableIdentifier], list[GetmlFKDef]]:
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

    def _next_identifier(
        n: str, max_idxs: dict[str, int], max_idxs_max: dict[str, int]
    ) -> GetmlTableIdentifier:
        i = max_idxs[n] + 1
        max_idxs[n] += 1
        if max_idxs[n] > max_idxs_max[n]:
            max_idxs_max[n] = max_idxs[n]
        return n, i

    # bfs itself

    _max_idxs: dict[str, int] = {k: -1 for k in schema}

    _nodes_this_layer: set[GetmlTableIdentifier] = {
        _next_identifier(target_table, _max_idxs, _max_idxs)
    }
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


def build_getml_datamodel(
    data_pd: dict[str, getml.data.DataFrame],
    nodes: set[GetmlTableIdentifier],
    edges: list[GetmlFKDef],
) -> getml.data.DataModel:
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
            t = t[len(prefix) :]
        out.append(t)
    return out


def evaluate_accuracy(
    targets: list[str], target_column: str, y_pred_prob: np.ndarray, y_true: pd.Series
) -> float:
    targets = strip_targets_prefix(targets, target_column)
    targets_idx_map = {v: i for i, v in enumerate(targets)}
    y_pred = np.argmax(y_pred_prob, 1)
    if y_true.apply(type).eq(str).all():
        y_true_np = y_true.map(targets_idx_map).to_numpy()
    else:
        y_true_np = y_true
    print(list(y_pred), y_pred.shape)
    print(list(y_true_np), y_true_np.shape)
    return np.mean(y_pred == y_true_np)


def main(dataset_name: str, max_depth: int):
    dataset = CTUDataset(dataset_name, data_dir="./datasets")
    data, _ = dataset.build_hetero_data()

    defaults = dataset.defaults
    schema = dataset.schema

    n_total = data[defaults.target_table].y.shape[0]
    data = T.RandomNodeSplit("train_rest", num_val=int(0.30 * n_total), num_test=0)(data)

    data_df = label_getml_roles(
        {n: t.df for n, t in dataset.db.table_dict.items()}, schema, defaults
    )

    # for tname, tbl in data_df.items():
    #     print(tname, tbl)

    # build split
    # split = getml.data.split.random(train=0.7, test=0.3)  # TODO use universal split
    split = pd.Series(data[defaults.target_table].train_mask.numpy())
    split = split.map({True: "train", False: "test"}).to_frame("split")
    split = getml.data.DataFrame.from_pandas(split, "split")["split"]

    container = getml.data.Container(population=data_df[TARGET_TABLE], split=split)
    container.add(**{k: v for k, v in data_df.items() if k != TARGET_TABLE})
    container.freeze()

    nodes, edges = bfs(schema, defaults.target_table, max_depth=max_depth)
    print(nodes)
    print(edges)
    dm = build_getml_datamodel(data_df, nodes, edges)
    print(dm)

    mapping = getml.preprocessors.Mapping()

    if defaults.task == TaskType.CLASSIFICATION:
        predictor = getml.predictors.XGBoostClassifier()
        fast_prop = getml.feature_learning.FastProp(
            loss_function=loss_functions.CrossEntropyLoss,
        )

    elif defaults.task == TaskType.REGRESSION:
        predictor = getml.predictors.XGBoostRegressor()
        fast_prop = getml.feature_learning.FastProp(
            loss_function=loss_functions.SquareLoss,
        )
    else:
        raise ValueError("unsupported task type")

    pipe = getml.pipeline.Pipeline(
        data_model=dm,
        preprocessors=[mapping],
        feature_learners=[fast_prop],
        # feature_selectors=[feature_selector],
        predictors=[predictor],
        share_selected_features=0.5,
    )

    pipe = pipe.fit(container.train)
    y_pred = pipe.predict(container.train)
    assert y_pred is not None

    # print(pipe.score(container.train))

    metrics = {}
    if defaults.task == TaskType.CLASSIFICATION:
        train_acc = evaluate_accuracy(
            pipe.targets,
            defaults.target_column,
            y_pred,
            data_df[defaults.target_table][split == "train"].to_pandas()[
                defaults.target_column
            ],
        )
        print("train_acc:", train_acc)
        metrics["best_train_acc"] = train_acc
    if defaults.task == TaskType.REGRESSION:
        target_vals: np.ndarray = (
            data_df[defaults.target_table][split == "train"]
            .to_pandas()[defaults.target_column]
            .apply(float)
            .to_numpy()
        )
        train_nrmse = (
            np.sqrt(((y_pred.squeeze() - target_vals) ** 2).mean()) / target_vals.mean()
        )
        print("train_nrmse:", train_nrmse)
        metrics["best_train_nrmse"] = train_nrmse

    y_pred = pipe.predict(container.test)

    # print(pipe.score(container.test))

    if defaults.task == TaskType.CLASSIFICATION:
        test_acc = evaluate_accuracy(
            pipe.targets,
            defaults.target_column,
            y_pred,
            data_df[defaults.target_table][split == "test"].to_pandas()[
                defaults.target_column
            ],
        )
        print("test_acc:", test_acc)
        metrics["best_val_acc"] = test_acc
    if defaults.task == TaskType.REGRESSION:
        target_vals: np.ndarray = (
            data_df[defaults.target_table][split == "test"]
            .to_pandas()[defaults.target_column]
            .apply(float)
            .to_numpy()
        )
        val_nrmse = (
            np.sqrt(((y_pred.squeeze() - target_vals) ** 2).mean()) / target_vals.mean()
        )
        print("val_nrmse:", val_nrmse)
        metrics["best_val_nrmse"] = val_nrmse

    return metrics


def run_experiment(
    tracking_uri: str,
    experiment_name: str,
    dataset: CTUDatasetName,
    run_name: str = None,
    log_dir: str = None,
    random_seed: int = RANDOM_SEED,
):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name=experiment_name)

    # getml_dir = f"./datasets/.getml"
    # Path(getml_dir).mkdir(parents=True, exist_ok=True)

    getml.engine.launch(
        launch_browser=False,
        project_directory=f"./datasets/{dataset}",
        allow_remote_ips=True,
    )
    getml.engine.set_project(dataset)

    time_str = datetime.now().strftime("%d-%m-%Y,%H:%M:%S")
    run_name = f"{dataset}_{time_str}" if run_name is None else run_name

    with mlflow.start_run(run_name=run_name) as parent_run:
        mlflow.set_tags(
            {
                MLFLOW_USER: "pelesjak",
                "Dataset": dataset,
            }
        )
        client = mlflow.tracking.MlflowClient(tracking_uri)

        for depth in range(2, 6):
            run = client.create_run(
                parent_run.info.experiment_id,
                run_name=f"{run_name}_{depth}",
                tags={
                    MLFLOW_USER: "pelesjak",
                    MLFLOW_PARENT_RUN_ID: parent_run.info.run_id,
                    "Dataset": dataset,
                },
            )
            client.log_param(run.info.run_id, "dataset", dataset)
            client.log_param(run.info.run_id, "max_depth", depth)

            try:
                metrics = main(dataset, depth)

                for m, v in metrics.items():
                    client.log_metric(run.info.run_id, m, v)
                mlflow.log_metrics({k: v for (k, v) in metrics.items()})
                break
            except Exception as e:
                print(traceback.format_exc())
                client.set_tag(run.info.run_id, "exception", str(e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET_NAME,
        choices=get_args(CTUDatasetName),
    )
    parser.add_argument("--experiment", type=str, default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default=None)

    args = parser.parse_args()
    print(args)

    run_experiment(
        "http://147.32.83.171:2222",
        args.experiment,
        args.dataset,
        args.run_name,
        args.log_dir,
    )
