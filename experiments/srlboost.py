import argparse
import collections
from datetime import datetime
import os, sys
import traceback


import random
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Optional, get_args
from typing import get_args as t_get_args

import mlflow
from mlflow.utils.mlflow_tags import MLFLOW_USER, MLFLOW_PARENT_RUN_ID

import numpy as np
import pandas as pd
import srlearn.base as srlearn_base
import torch
import torch_geometric.transforms as T

from torchmetrics import (
    Accuracy,
    F1Score,
    MeanAbsoluteError,
    MeanSquaredError,
    MeanSquaredLogError,
    NormalizedRootMeanSquaredError,
    R2Score,
    AUROC,
)
from lightning.pytorch import seed_everything

from slugify import slugify
from srlearn import Background
from srlearn.base import FileSystem
from srlearn.database import Database
from srlearn.rdn import BoostedRDNClassifier, BoostedRDNRegressor
from srlearn.system_manager import BoostSRLFiles
from tqdm.auto import tqdm

sys.path.append(os.getcwd())
from db_transformer.data import CTUDatasetName, CTUDataset, TaskType
from db_transformer.schema.columns import CategoricalColumnDef
from db_transformer.schema.schema import ForeignKeyDef, Schema


DEFAULT_DATASET_NAME: CTUDatasetName = "CORA"

DEFAULT_EXPERIMENT_NAME = "pelesjak-deep-db-tests"

RANDOM_SEED = 42


def _sanitize_value(value) -> str:
    return slugify(str(value)).replace("-", "_")


def get_fact_def_for_feature(
    table_name: str, column_name: str, schema: Schema
) -> tuple[str, str, str] | None:
    if not isinstance(schema[table_name].columns[column_name], CategoricalColumnDef):
        return

    return f"{table_name}_{column_name}", table_name, f"{table_name}_{column_name}"


def get_fact_def_for_target(
    table_name: str, column_name: str, value: str, schema: Schema
) -> tuple[str, str, None]:
    assert isinstance(schema[table_name].columns[column_name], CategoricalColumnDef)

    return f"{table_name}_{column_name}_{_sanitize_value(value)}", table_name, None


def get_fact_name(fact_def: tuple[str, Any, Any] | None) -> str | None:
    if fact_def is None:
        return None

    fact_name, _, _ = fact_def
    return fact_name


def feature_to_fact(
    table_name: str, column_name: str, df: pd.DataFrame, schema: Schema
) -> Iterable[str]:
    fact_name = get_fact_name(get_fact_def_for_feature(table_name, column_name, schema))

    if fact_name is None:
        return

    pk_all = sorted(schema[table_name].get_primary_key())

    df = df[[*pk_all, column_name]]

    for _, row in df.iterrows():
        idx = "_".join([str(row[pk]) for pk in pk_all])
        val = row[column_name]

        row_name = _sanitize_value(f"{table_name}_{idx}")
        feature_value_name = _sanitize_value(f"{fact_name}_{val}")

        yield f"{fact_name}({row_name}, {feature_value_name})."


def target_to_fact(
    table_name: str, column_name: str, target_value: str, df: pd.DataFrame, schema: Schema
) -> Iterable[str]:
    fact_name = get_fact_name(
        get_fact_def_for_target(table_name, column_name, target_value, schema)
    )

    assert fact_name is not None

    pk_all = sorted(schema[table_name].get_primary_key())

    df = df[[*pk_all, column_name]]

    for _, row in df.iterrows():
        idx = "_".join([str(row[pk]) for pk in pk_all])
        row_name = _sanitize_value(f"{table_name}_{idx}")
        yield f"{fact_name}({row_name})."


def get_fact_def_for_foreign_key(
    table_name: str, fk_def: ForeignKeyDef
) -> tuple[str, str, str]:
    fk_all = sorted(fk_def.columns)
    fk_name = "_".join(fk_all)
    fact_name = f"{table_name}_{fk_name}_{fk_def.ref_table}"
    return fact_name, table_name, fk_def.ref_table


def foreign_key_to_fact(
    table_name: str, fk_def: ForeignKeyDef, df: pd.DataFrame, schema: Schema
) -> Iterable[str]:
    pk_all = sorted(schema[table_name].get_primary_key())
    fk_all = sorted(fk_def.columns)

    df = df[[*pk_all, *fk_all]]

    fact_name = get_fact_name(get_fact_def_for_foreign_key(table_name, fk_def))
    assert fact_name is not None

    for _, row in df.iterrows():
        idx_left = "_".join([str(row[pk]) for pk in pk_all])
        idx_right = "_".join([str(row[fk]) for fk in fk_all])

        row_left_name = _sanitize_value(f"{table_name}_{idx_left}")
        row_right_name = _sanitize_value(f"{fk_def.ref_table}_{idx_right}")

        yield f"{fact_name}({row_left_name}, {row_right_name})."


def get_all_feature_fact_defs(
    schema: Schema, target: tuple[str, str]
) -> Iterable[tuple[str, str, str]]:
    for table_name, table_def in schema.items():
        for column_name in table_def.columns:
            if (table_name, column_name) != target:
                fact_def = get_fact_def_for_feature(table_name, column_name, schema)
                if fact_def is not None:
                    yield fact_def


def get_all_fk_fact_defs(schema: Schema) -> Iterable[tuple[str, str, str]]:
    for table_name, table_def in schema.items():
        for fk_def in table_def.foreign_keys:
            yield get_fact_def_for_foreign_key(table_name, fk_def)


def get_all_fact_defs(
    schema: Schema, target: tuple[str, str], target_values: list[str]
) -> Iterable[tuple[str, str, str | None]]:
    yield from get_all_feature_fact_defs(schema, target)
    yield from get_all_fk_fact_defs(schema)

    for target_value in target_values:
        yield get_fact_def_for_target(*target, target_value, schema)


def get_all_fact_names(
    schema: Schema, target: tuple[str, str], target_values: list[str]
) -> list[str]:
    return [
        fact_name for fact_name, _, _ in get_all_fact_defs(schema, target, target_values)
    ]


def map_all_values_to_idx(
    dfs: dict[str, pd.DataFrame], exclude: set[tuple[str, str]] | None = None
) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}

    if exclude is None:
        exclude = set()

    for table_name, df in dfs.items():
        df = df.copy()

        for column_name in df.columns:
            if (table_name, column_name) in exclude:
                continue

            the_map = {value: i for i, value in enumerate(df[column_name].unique())}
            df[column_name] = df[column_name].map(the_map)

        out[table_name] = df
    return out


def build_dataset(
    schema: Schema,
    dfs: dict[str, pd.DataFrame],
    target: tuple[str, str],
    target_value: str,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    all_values_to_idx: bool = False,
) -> tuple[Database, Database]:
    train = Database()
    test = Database()

    if all_values_to_idx:
        dfs = map_all_values_to_idx(dfs, exclude={target})

    # build modes
    modes: list[str] = []

    for fact_name, row_name, feature_name in get_all_feature_fact_defs(schema, target):
        modes.append(f"{fact_name}(+{row_name}, #{feature_name}).")
    for fact_name, row_left_name, row_right_name in get_all_fk_fact_defs(schema):
        modes.append(f"{fact_name}(+{row_left_name}, -{row_right_name}).")
        modes.append(f"{fact_name}(-{row_left_name}, +{row_right_name}).")

    fact_name, row_name, _ = get_fact_def_for_target(*target, target_value, schema)
    modes.append(f"{fact_name}(+{row_name}).")

    train.modes = modes
    test.modes = modes

    # build facts
    facts: list[str] = []
    for table_name, table_def in schema.items():
        for column_name, _ in table_def.columns.items():
            if (table_name, column_name) == target:
                continue

            facts.extend(feature_to_fact(table_name, column_name, dfs[table_name], schema))

        for fk_def in table_def.foreign_keys:
            facts.extend(foreign_key_to_fact(table_name, fk_def, dfs[table_name], schema))

    train.facts = facts
    test.facts = facts

    # build target
    train_pos: list[str] = []
    train_neg: list[str] = []
    test_pos: list[str] = []
    test_neg: list[str] = []

    target_df = dfs[target[0]]

    train_df = target_df[pd.Series(train_mask)]
    train_df_pos = train_df[train_df[target[1]] == target_value]
    train_df_neg = train_df[train_df[target[1]] != target_value]
    train_pos.extend(
        target_to_fact(target[0], target[1], target_value, train_df_pos, schema)
    )
    train_neg.extend(
        target_to_fact(target[0], target[1], target_value, train_df_neg, schema)
    )

    test_df = target_df[pd.Series(val_mask)]
    test_df_pos = test_df[test_df[target[1]] == target_value]
    test_df_neg = test_df[test_df[target[1]] != target_value]
    test_pos.extend(target_to_fact(target[0], target[1], target_value, test_df_pos, schema))
    test_neg.extend(target_to_fact(target[0], target[1], target_value, test_df_neg, schema))

    if len(train_pos) == 0 and len(test_pos) > 0:
        train_pos = [test_pos[0]]
    elif len(test_pos) == 0 and len(train_pos) > 0:
        test_pos = [train_pos[0]]
    elif len(train_pos) == 0 and len(test_pos) == 0:
        raise ValueError(
            f"Literally no positive examples, what??? (target: {target_value})"
        )

    train.pos = train_pos
    train.neg = train_neg
    test.pos = test_pos
    test.neg = test_neg

    return train, test


class CustomFileSystem(FileSystem):
    def __init__(self, dataset_name: str, target_value: str):
        jar_root = Path(srlearn_base.__file__).parent

        # Allocate a location where data can safely be stored.
        data = Path(f"./datasets/{dataset_name}") / FileSystem.boostsrl_data_directory
        data.mkdir(exist_ok=True)

        dataset_dir = data / "dataset"
        dataset_dir.mkdir(exist_ok=True)

        directory = dataset_dir / slugify(str(target_value))
        directory.mkdir(exist_ok=True)

        self.files = BoostSRLFiles(directory, jar_root)
        self.files.TRAIN_DIR.mkdir(exist_ok=True)
        self.files.TEST_DIR.mkdir(exist_ok=True)


def wrap_class_with_custom_file_system(dataset_name: str, target_value: str):
    # @wraps(cls)
    class CustomTrainer(BoostedRDNClassifier):
        def _check_params(self):
            super()._check_params()
            self.file_system = CustomFileSystem(dataset_name, target_value)

    return CustomTrainer


def get_metrics(
    task_type: TaskType, num_classes: Optional[int] = None
) -> dict[str, Callable[[np.ndarray, np.ndarray], float]]:
    metrics = {}
    if task_type == TaskType.CLASSIFICATION:
        assert (
            num_classes is not None
        ), "num_classes is required to initialize metrics for classification task"
        task = "binary" if num_classes == 2 else "multiclass"
        metrics["acc"] = Accuracy(task=task, num_classes=num_classes, average="micro")
        metrics["f1_micro"] = F1Score(task=task, num_classes=num_classes, average="micro")
        metrics["f1_macro"] = F1Score(task=task, num_classes=num_classes, average="macro")
        metrics["auroc"] = AUROC(task=task, num_classes=num_classes, average="macro")

    elif task_type == TaskType.REGRESSION:
        metrics["mae"] = MeanAbsoluteError()
        metrics["mse"] = MeanSquaredError()
        metrics["msle"] = MeanSquaredLogError()
        metrics["nrmse"] = NormalizedRootMeanSquaredError(normalization="range")

    else:
        raise ValueError("unsupported task type")

    return metrics


def run(dataset_name: str, seed: int = RANDOM_SEED) -> dict[str, Any]:
    dataset = CTUDataset(dataset_name)

    schema = dataset.schema
    defaults = dataset.defaults

    assert defaults.task == TaskType.CLASSIFICATION

    dfs = {n: t.df for n, t in dataset.db.table_dict.items()}
    data, _ = dataset.build_hetero_data()

    seed_everything(seed)
    n_total = data[defaults.target_table].y.shape[0]
    data = T.RandomNodeSplit("train_rest", num_val=int(0.30 * n_total), num_test=0)(data)

    target = (defaults.target_table, defaults.target_column)
    train_mask = data[defaults.target_table].train_mask.numpy()
    val_mask = data[defaults.target_table].val_mask.numpy()

    # drop na values from target and from masks
    train_mask = train_mask[
        ~dfs[defaults.target_table][defaults.target_column].isna().to_numpy()
    ]
    val_mask = val_mask[
        ~dfs[defaults.target_table][defaults.target_column].isna().to_numpy()
    ]
    dfs[defaults.target_table].dropna(
        subset=[defaults.target_column], inplace=True, ignore_index=True
    )

    target_values = dfs[defaults.target_table][defaults.target_column].unique().tolist()

    train_results_per_class: collections.OrderedDict[str, np.ndarray] = (
        collections.OrderedDict()
    )
    test_results_per_class: collections.OrderedDict[str, np.ndarray] = (
        collections.OrderedDict()
    )

    assert len(target_values) >= 2

    print(f"Target values: {target_values}")

    metrics = get_metrics(defaults.task, num_classes=len(target_values))

    for target_value in (progress := tqdm(target_values)):
        progress.set_postfix(target=target_value)
        train, test = build_dataset(
            schema,
            dfs,
            target,
            target_value,
            train_mask,
            val_mask,
            all_values_to_idx=True,
        )

        target_train_series = dfs[defaults.target_table][defaults.target_column][
            pd.Series(train_mask)
        ]
        train_mask_pos = target_train_series == target_value
        order_indices_train = np.concatenate(
            [np.where(train_mask_pos)[0], np.where(~train_mask_pos)[0]]
        )

        target_val_series = dfs[defaults.target_table][defaults.target_column][
            pd.Series(val_mask)
        ]
        val_mask_pos = target_val_series == target_value
        order_indices_val = np.concatenate(
            [np.where(val_mask_pos)[0], np.where(~val_mask_pos)[0]]
        )

        bk = Background(modes=train.modes)

        custom_trainer_cls = wrap_class_with_custom_file_system(dataset_name, target_value)

        clf = custom_trainer_cls(
            solver="SRLBoost",
            background=bk,
            target=get_fact_name(get_fact_def_for_target(*target, target_value, schema)),
        )
        clf.fit(train)

        p = clf.predict_proba(train)
        train_results_per_class[target_value] = p[np.argsort(order_indices_train)]

        p = clf.predict_proba(test)
        test_results_per_class[target_value] = p[np.argsort(order_indices_val)]

        # pred = np.greater(p, clf.threshold_)
    if len(target_values) == 2:
        target_value = target_values[1]

        y_true: pd.Series = (
            dfs[defaults.target_table][defaults.target_column] == target_value
        )
        y_true_train = y_true[train_mask].astype(int).to_numpy()
        y_true_test = y_true[val_mask].astype(int).to_numpy()

        y_pred_train = np.greater(
            train_results_per_class[target_value], clf.threshold_
        ).astype(int)
        y_pred_test = np.greater(
            test_results_per_class[target_value], clf.threshold_
        ).astype(int)
    else:
        cls_index = {value: i for i, value in enumerate(target_values)}
        y_true = (
            dfs[defaults.target_table][defaults.target_column].map(cls_index).to_numpy()
        )
        y_true_train = y_true[train_mask]
        y_true_test = y_true[val_mask]

        y_pred_train = np.argmax(np.stack(list(train_results_per_class.values()), -1), -1)
        y_pred_test = np.argmax(np.stack(list(test_results_per_class.values()), -1), -1)

    metrics_report = {}
    for mname, metric in metrics.items():
        metric_train = metric(
            torch.from_numpy(y_pred_train), torch.from_numpy(y_true_train)
        )
        metric_test = metric(torch.from_numpy(y_pred_test), torch.from_numpy(y_true_test))
        metrics_report[f"best_train_{mname}"] = metric_train
        metrics_report[f"best_val_{mname}"] = metric_test
    print(metrics_report)
    return metrics_report


def run_experiment(
    tracking_uri: str,
    experiment_name: str,
    dataset: CTUDatasetName,
    run_name: str = None,
    log_dir: str = None,
    random_seed: int = RANDOM_SEED,
):

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name=experiment_name)

    time_str = datetime.now().strftime("%d-%m-%Y,%H:%M:%S")
    run_name = f"{dataset}_{time_str}" if run_name is None else run_name

    with mlflow.start_run(run_name=run_name) as parent_run:
        mlflow.set_tags(
            {
                MLFLOW_USER: "pelesjak",
                "Dataset": dataset,
            }
        )
        mlflow.log_params(dict(dataset=dataset))

        try:
            metrics = run(dataset, random_seed)
            mlflow.log_metrics(metrics)
        except Exception as e:
            print(traceback.format_exc())
            mlflow.set_tag("exception", str(e))
            mlflow.end_run("FAILED")


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
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)

    args = parser.parse_args()

    # run(args.dataset)
    print(args)

    run_experiment(
        "http://147.32.83.171:2222",
        args.experiment,
        args.dataset,
        args.run_name,
        args.log_dir,
        args.seed,
    )
