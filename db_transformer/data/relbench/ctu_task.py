import sys, os

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from db_transformer.data.relbench.ctu_repository_defauts import CTU_REPOSITORY_DEFAULTS

from relbench.data import BaseTask, Database, Table
from relbench.metrics import accuracy, average_precision, f1, mae, rmse, roc_auc

if TYPE_CHECKING:
    from db_transformer.data.relbench.ctu_dataset import CTUDataset


class CTUTask(BaseTask):

    name = "ctu-task"

    def __init__(self, dataset: "CTUDataset", time_split: bool = False):
        self.defaults = CTU_REPOSITORY_DEFAULTS[dataset.name]
        time_col = self.defaults.timestamp_column
        if time_split and time_col is not None:
            df = dataset.db.table_dict[self.defaults.target_table].df

            sorted = df.sort_values(by=[time_col])

            train, validate, test = np.split(
                sorted, [int(0.6 * len(df)), int(0.8 * len(df))]
            )

        metrics = []

        super().__init__(dataset, pd.Timedelta(days=0), metrics)

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        target_table = db.table_dict[self.defaults.target_table]
        print(target_table.df.dtypes)
        # print(target_table.pkey_col)

        target_cols = (
            [target_table.pkey_col, self.defaults.target_column]
            if target_table.pkey_col is not None
            else [self.defaults.target_column]
        )
        df = target_table.df[target_cols].copy(deep=True)

        # TODO: make random splits

        return Table(df, {target_table.pkey_col: self.defaults.target_table})

    @property
    def train_table(self) -> Table:
        """Returns the train table for a task."""

        return self.make_table(self.dataset.db, None)

    @property
    def val_table(self) -> Table:
        r"""Returns the val table for a task."""

        return self.make_table(self.dataset.db, None)

    @property
    def test_table(self) -> Table:
        r"""Returns the test table for a task."""

        return self.make_table(self.dataset.db, None)
