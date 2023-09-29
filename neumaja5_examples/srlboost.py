import argparse
import collections
import os
import random
import uuid
from pathlib import Path
from typing import Any, Iterable, Literal
from typing import get_args as t_get_args

import lovely_tensors as lt
import mlflow
import numpy as np
import pandas as pd
import srlearn.base as srlearn_base
import torch
import torch_geometric.transforms as T
from db_transformer.data.dataset_defaults.fit_dataset_defaults import FIT_DATASET_DEFAULTS
from db_transformer.data.fit_dataset import FITRelationalDataset
from db_transformer.data.utils.heterodata_builder import HeteroDataBuilder
from db_transformer.schema.columns import CategoricalColumnDef
from db_transformer.schema.schema import ForeignKeyDef, Schema
from slugify import slugify
from srlearn import Background
from srlearn.base import FileSystem
from srlearn.database import Database
from srlearn.rdn import BoostedRDNClassifier
from srlearn.system_manager import BoostSRLFiles
from tqdm.auto import tqdm

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


DEFAULT_DATASET_NAME: DatasetType = 'imdb_ijs'


def _sanitize_value(value) -> str:
    return slugify(str(value)).replace('-', '_')


def get_fact_def_for_feature(table_name: str, column_name: str, schema: Schema) -> tuple[str, str, str] | None:
    if not isinstance(schema[table_name].columns[column_name], CategoricalColumnDef):
        return

    return f"{table_name}_{column_name}", table_name, f"{table_name}_{column_name}"


def get_fact_def_for_target(table_name: str, column_name: str, value: str, schema: Schema) -> tuple[str, str, None]:
    assert isinstance(schema[table_name].columns[column_name], CategoricalColumnDef)

    return f"{table_name}_{column_name}_{_sanitize_value(value)}", table_name, None


def get_fact_name(fact_def: tuple[str, Any, Any] | None) -> str | None:
    if fact_def is None:
        return None

    fact_name, _, _ = fact_def
    return fact_name


def feature_to_fact(table_name: str, column_name: str, df: pd.DataFrame, schema: Schema) -> Iterable[str]:
    fact_name = get_fact_name(get_fact_def_for_feature(table_name, column_name, schema))

    if fact_name is None:
        return

    pk_all = sorted(schema[table_name].get_primary_key())

    df = df[[*pk_all, column_name]]

    for row in df.itertuples():
        idx = '_'.join([str(getattr(row, pk)) for pk in pk_all])
        val = getattr(row, column_name)

        row_name = _sanitize_value(f"{table_name}_{idx}")
        feature_value_name = _sanitize_value(f"{fact_name}_{val}")

        yield f"{fact_name}({row_name}, {feature_value_name})."


def target_to_fact(table_name: str, column_name: str, target_value: str, df: pd.DataFrame, schema: Schema) -> Iterable[str]:
    fact_name = get_fact_name(get_fact_def_for_target(table_name, column_name, target_value, schema))

    assert fact_name is not None

    pk_all = sorted(schema[table_name].get_primary_key())

    df = df[[*pk_all, column_name]]

    for row in df.itertuples():
        idx = '_'.join([str(getattr(row, pk)) for pk in pk_all])
        row_name = _sanitize_value(f"{table_name}_{idx}")
        yield f"{fact_name}({row_name})."


def get_fact_def_for_foreign_key(table_name: str, fk_def: ForeignKeyDef) -> tuple[str, str, str]:
    fk_all = sorted(fk_def.columns)
    fk_name = '_'.join(fk_all)
    fact_name = f"{table_name}_{fk_name}_{fk_def.ref_table}"
    return fact_name, table_name, fk_def.ref_table


def foreign_key_to_fact(table_name: str, fk_def: ForeignKeyDef, df: pd.DataFrame, schema: Schema) -> Iterable[str]:
    pk_all = sorted(schema[table_name].get_primary_key())
    fk_all = sorted(fk_def.columns)

    df = df[[*pk_all, *fk_all]]

    fact_name = get_fact_name(get_fact_def_for_foreign_key(table_name, fk_def))
    assert fact_name is not None

    for row in df.itertuples():
        idx_left = '_'.join([str(getattr(row, pk)) for pk in pk_all])
        idx_right = '_'.join([str(getattr(row, fk)) for fk in fk_all])

        row_left_name = _sanitize_value(f"{table_name}_{idx_left}")
        row_right_name = _sanitize_value(f"{fk_def.ref_table}_{idx_right}")

        yield f"{fact_name}({row_left_name}, {row_right_name})."


def get_all_feature_fact_defs(schema: Schema, target: tuple[str, str]) -> Iterable[tuple[str, str, str]]:
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


def get_all_fact_defs(schema: Schema, target: tuple[str, str], target_values: list[str]) -> Iterable[tuple[str, str, str | None]]:
    yield from get_all_feature_fact_defs(schema, target)
    yield from get_all_fk_fact_defs(schema)

    for target_value in target_values:
        yield get_fact_def_for_target(*target, target_value, schema)


def get_all_fact_names(schema: Schema, target: tuple[str, str], target_values: list[str]) -> list[str]:
    return [fact_name for fact_name, _, _ in get_all_fact_defs(schema, target, target_values)]


def map_all_values_to_idx(dfs: dict[str, pd.DataFrame], exclude: set[tuple[str, str]] | None = None) -> dict[str, pd.DataFrame]:
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


def build_dataset(schema: Schema, dfs: dict[str, pd.DataFrame], target: tuple[str, str], target_value: str, train_mask: np.ndarray, val_mask: np.ndarray, all_values_to_idx: bool = False) -> tuple[Database, Database]:
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
    train_pos.extend(target_to_fact(target[0], target[1], target_value, train_df_pos, schema))
    train_neg.extend(target_to_fact(target[0], target[1], target_value, train_df_neg, schema))

    test_df = target_df[pd.Series(val_mask)]
    test_df_pos = test_df[test_df[target[1]] == target_value]
    test_df_neg = test_df[test_df[target[1]] != target_value]
    test_pos.extend(target_to_fact(target[0], target[1], target_value, test_df_pos, schema))
    test_neg.extend(target_to_fact(target[0], target[1], target_value, test_df_neg, schema))

    train.pos = train_pos
    train.neg = train_neg
    test.pos = test_pos
    test.neg = test_neg

    return train, test


class CustomFileSystem(FileSystem):
    def __init__(self, dataset_name: str, target_value: str):
        jar_root = Path(srlearn_base.__file__).parent

        # Allocate a location where data can safely be stored.
        data = Path('.') / FileSystem.boostsrl_data_directory
        data.mkdir(exist_ok=True)

        dataset_dir = data / dataset_name
        dataset_dir.mkdir(exist_ok=True)

        directory = dataset_dir / slugify(str(target_value))
        directory.mkdir(exist_ok=True)

        self.files = BoostSRLFiles(directory, jar_root)
        self.files.TRAIN_DIR.mkdir(exist_ok=True)
        self.files.TEST_DIR.mkdir(exist_ok=True)


def wrap_class_with_custom_file_system(cls, dataset_name: str, target_value: str):
    # @wraps(cls)
    class CustomTrainer(cls):
        def _check_params(self):
            super()._check_params()
            self.file_system = CustomFileSystem(dataset_name, target_value)

    return CustomTrainer


def run(dataset_name: str) -> dict[str, Any]:
    with FITRelationalDataset.create_remote_connection(dataset_name) as conn:
        defaults = FIT_DATASET_DEFAULTS[dataset_name]

        schema = FITRelationalDataset.create_schema_analyzer(
            dataset_name, conn, verbose=True).guess_schema()

        builder = HeteroDataBuilder(conn,
                                    schema,
                                    target_table=defaults.target_table,
                                    target_column=defaults.target_column,
                                    separate_target=True,
                                    create_reverse_edges=True,
                                    fillna_with=0.0)

        dfs = builder._get_dataframes_raw()
        data, _ = builder.build(with_column_names=False)

        n_total = data[defaults.target_table].x.shape[0]
        T.RandomNodeSplit('train_rest', num_val=int(0.30 * n_total), num_test=0)(data)

    target_values = dfs[defaults.target_table][defaults.target_column].unique().tolist()

    target = (defaults.target_table, defaults.target_column)
    train_mask = data[defaults.target_table].train_mask.numpy()
    val_mask = data[defaults.target_table].val_mask.numpy()

    train_results_per_class: collections.OrderedDict[str, np.ndarray] = collections.OrderedDict()
    test_results_per_class: collections.OrderedDict[str, np.ndarray] = collections.OrderedDict()

    assert len(target_values) >= 2

    if len(target_values) == 2:
        target_values = [tv for tv in target_values if tv]
        assert len(target_values) > 0
        target_values = target_values[:1]

    for target_value in tqdm(target_values):
        train, test = build_dataset(schema, dfs, target, target_value, train_mask, val_mask, all_values_to_idx=True)

        target_train_series = dfs[defaults.target_table][defaults.target_column][pd.Series(train_mask)]
        train_mask_pos = target_train_series == target_value
        order_indices_train = np.concatenate([np.where(train_mask_pos)[0], np.where(~train_mask_pos)[0]])

        target_val_series = dfs[defaults.target_table][defaults.target_column][pd.Series(val_mask)]
        val_mask_pos = target_val_series == target_value
        order_indices_val = np.concatenate([np.where(val_mask_pos)[0], np.where(~val_mask_pos)[0]])

        bk = Background(modes=train.modes)

        custom_trainer_cls = wrap_class_with_custom_file_system(BoostedRDNClassifier, dataset_name, target_value)

        clf = custom_trainer_cls(
            solver='SRLBoost',
            background=bk,
            target=get_fact_name(get_fact_def_for_target(*target, target_value, schema)),
        )
        clf.fit(train)

        p = clf.predict_proba(train)
        train_results_per_class[target_value] = p[np.argsort(order_indices_train)]

        p = clf.predict_proba(test)
        test_results_per_class[target_value] = p[np.argsort(order_indices_val)]

        # pred = np.greater(p, clf.threshold_)

    if len(target_values) == 1:
        target_value = target_values[0]

        y_true = (dfs[defaults.target_table][defaults.target_column] == target_value).to_numpy()
        y_true_train = y_true[train_mask]
        y_true_test = y_true[val_mask]

        y_pred_train = np.greater(train_results_per_class[target_value], clf.threshold_)
        y_pred_test = np.greater(test_results_per_class[target_value], clf.threshold_)
    else:
        cls_index = {value: i for i, value in enumerate(target_values)}
        y_true = dfs[defaults.target_table][defaults.target_column].map(cls_index).to_numpy()
        y_true_train = y_true[train_mask]
        y_true_test = y_true[val_mask]

        y_pred_train = np.argmax(np.stack(list(train_results_per_class.values()), -1), -1)
        y_pred_test = np.argmax(np.stack(list(test_results_per_class.values()), -1), -1)

    train_acc = np.mean(y_pred_train == y_true_train)
    val_acc = np.mean(y_pred_test == y_true_test)

    result = dict(
        train_acc=train_acc,
        val_acc=val_acc,
        best_train_acc=train_acc,
        best_val_acc=val_acc,
    )
    print(result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices=t_get_args(DatasetType))
    parser.add_argument("--mlflow", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args([DEFAULT_DATASET_NAME] if '__file__' not in locals() else None)

    dataset_name: DatasetType = args.dataset
    do_mlflow: bool = args.mlflow

    if do_mlflow:
        os.environ['MLFLOW_TRACKING_URI'] = 'http://147.32.83.171:2222'
        mlflow.set_experiment("deep_rl_learning NEW - neumaja5")
        mlflow.pytorch.autolog()

        file_name = os.path.basename(__file__)
        with mlflow.start_run(run_name=f"{file_name} - {dataset_name} - {uuid.uuid4()}"):
            mlflow.set_tag('dataset', dataset_name)
            mlflow.set_tag('Model Source', file_name)

            try:
                result = run(dataset_name)
            except Exception as ex:
                mlflow.set_tag('exception', str(ex))
                raise ex

            for k, v in result.items():
                mlflow.log_metric(k, v)
    else:
        run(dataset_name)
