from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, Literal, Optional, Tuple, Union

from db_transformer.data.dataset_defaults.fit_dataset_fixes import fix_citeseer_schema
from db_transformer.db.distinct_cnt_retrieval import DBDistinctCounter
from db_transformer.db.schema_autodetect import BuiltinDBDistinctCounter
from db_transformer.schema.schema import Schema


class TaskType(Enum):
    CLASSIFICATION = 1
    REGRESSION = 2


# fmt:off
FITDatasetName = Literal[
    "Accidents", "Airline", "Atherosclerosis", "Basketball_women", "Bupa", "Carcinogenesis",
    "Chess", "CiteSeer", "ConsumerExpenditures", "CORA", "CraftBeer", "Credit", "cs",
    "Dallas", "DCG", "Dunur", "Elti", "ErgastF1", "Facebook", "financial", "ftp", "geneea",
    "genes", "Hepatitis_std", "Hockey", "imdb_ijs", "imdb_MovieLens", "KRK", "legalActs",
    "medical", "Mondial", "Mooney_Family", "MuskSmall", "mutagenesis", "nations", "NBA",
    "NCAA", "Pima", "PremierLeague", "PTE", "PubMed_Diabetes", "Same_gen", "SAP", "SAT",
    "Shakespeare", "Student_loan", "Toxicology", "tpcc", "tpcd", "tpcds", "trains",
    "university", "UTube", "UW_std", "VisualGenome", "voc", "WebKP", "world"
]
# fmt:on


@dataclass
class FITDatasetDefaults:
    target_table: str
    target_column: str
    task: TaskType
    schema_fixer: Optional[Callable[[Schema], None]] = None
    db_distinct_counter: Union[DBDistinctCounter, BuiltinDBDistinctCounter] = "db_distinct"
    force_collation: Optional[str] = None

    @property
    def target(self) -> Tuple[str, str]:
        return self.target_table, self.target_column


FIT_DATASET_DEFAULTS: Dict[FITDatasetName, FITDatasetDefaults] = {
    "Accidents": FITDatasetDefaults(
        target_table="nesreca", target_column="klas_nesreca", task=TaskType.CLASSIFICATION
    ),
    "AdventureWorks2014": FITDatasetDefaults(
        target_table="SalesOrderHeader",
        target_column="TotalDue",
        task=TaskType.REGRESSION,
    ),
    "Airline": FITDatasetDefaults(
        target_table="On_Time_On_Time_Performance_2016_1",
        target_column="ArrDel15",
        task=TaskType.CLASSIFICATION,
    ),
    "Atherosclerosis": FITDatasetDefaults(
        target_table="Death", target_column="PRICUMR", task=TaskType.CLASSIFICATION
    ),
    "Basketball_men": FITDatasetDefaults(
        target_table="teams", target_column="rank", task=TaskType.REGRESSION
    ),
    "Basketball_women": FITDatasetDefaults(
        target_table="teams", target_column="playoff", task=TaskType.CLASSIFICATION
    ),
    "Biodegradability": FITDatasetDefaults(
        target_table="molecule", target_column="activity", task=TaskType.REGRESSION
    ),
    "Bupa": FITDatasetDefaults(
        target_table="bupa", target_column="arg2", task=TaskType.CLASSIFICATION
    ),
    "Carcinogenesis": FITDatasetDefaults(
        target_table="canc", target_column="class", task=TaskType.CLASSIFICATION
    ),
    "ccs": FITDatasetDefaults(
        target_table="transactions_1k", target_column="Price", task=TaskType.REGRESSION
    ),
    "CDESchools": FITDatasetDefaults(
        target_table="satscores", target_column="PctGE1500", task=TaskType.REGRESSION
    ),
    "Chess": FITDatasetDefaults(
        target_table="game", target_column="game_result", task=TaskType.CLASSIFICATION
    ),
    "CiteSeer": FITDatasetDefaults(
        target_table="paper",
        target_column="class_label",
        task=TaskType.CLASSIFICATION,
        schema_fixer=fix_citeseer_schema,
    ),
    "classicmodels": FITDatasetDefaults(
        target_table="payments", target_column="amount", task=TaskType.REGRESSION
    ),
    "ConsumerExpenditures": FITDatasetDefaults(
        target_table="EXPENDITURES", target_column="GIFT", task=TaskType.CLASSIFICATION
    ),
    "CORA": FITDatasetDefaults(
        target_table="paper", target_column="class_label", task=TaskType.CLASSIFICATION
    ),
    "Countries": FITDatasetDefaults(
        target_table="target", target_column="2012", task=TaskType.REGRESSION
    ),
    "CraftBeer": FITDatasetDefaults(
        target_table="breweries", target_column="state", task=TaskType.CLASSIFICATION
    ),
    "Credit": FITDatasetDefaults(
        target_table="member", target_column="region_no", task=TaskType.CLASSIFICATION
    ),
    "cs": FITDatasetDefaults(
        target_table="target_churn",
        target_column="target_churn",
        task=TaskType.CLASSIFICATION,
    ),
    "Dallas": FITDatasetDefaults(
        target_table="incidents",
        target_column="subject_statuses",
        task=TaskType.CLASSIFICATION,
    ),
    "DCG": FITDatasetDefaults(
        target_table="sentences", target_column="class", task=TaskType.CLASSIFICATION
    ),
    "Dunur": FITDatasetDefaults(
        target_table="target", target_column="is_dunur", task=TaskType.CLASSIFICATION
    ),
    "Elti": FITDatasetDefaults(
        target_table="target", target_column="is_elti", task=TaskType.CLASSIFICATION
    ),
    "employee": FITDatasetDefaults(
        target_table="salaries", target_column="salary", task=TaskType.REGRESSION
    ),
    "ErgastF1": FITDatasetDefaults(
        target_table="target", target_column="win", task=TaskType.CLASSIFICATION
    ),
    "Facebook": FITDatasetDefaults(
        target_table="feat", target_column="gender1", task=TaskType.CLASSIFICATION
    ),
    "financial": FITDatasetDefaults(
        target_table="loan", target_column="status", task=TaskType.CLASSIFICATION
    ),
    "FNHK": FITDatasetDefaults(
        target_table="pripady",
        target_column="Delka_hospitalizace",
        task=TaskType.REGRESSION,
    ),
    "ftp": FITDatasetDefaults(
        target_table="session", target_column="gender", task=TaskType.CLASSIFICATION
    ),
    "geneea": FITDatasetDefaults(
        target_table="hl_hlasovani",
        target_column="vysledek",
        task=TaskType.CLASSIFICATION,
        force_collation="utf8mb3_unicode_ci",
    ),
    "genes": FITDatasetDefaults(
        target_table="Classification",
        target_column="Localization",
        task=TaskType.CLASSIFICATION,
    ),
    "GOSales": FITDatasetDefaults(
        target_table="go_1k", target_column="Quantity", task=TaskType.REGRESSION
    ),
    "Grants": FITDatasetDefaults(
        target_table="awards", target_column="award_amount", task=TaskType.REGRESSION
    ),
    "Hepatitis_std": FITDatasetDefaults(
        target_table="dispat", target_column="Type", task=TaskType.CLASSIFICATION
    ),
    "Hockey": FITDatasetDefaults(
        target_table="Master", target_column="shootCatch", task=TaskType.CLASSIFICATION
    ),
    "imdb_ijs": FITDatasetDefaults(
        target_table="actors",
        target_column="gender",
        task=TaskType.CLASSIFICATION,
        db_distinct_counter="fetchall_unidecode_strip_ci",
    ),
    "imdb_MovieLens": FITDatasetDefaults(
        target_table="users", target_column="u_gender", task=TaskType.CLASSIFICATION
    ),
    "KRK": FITDatasetDefaults(
        target_table="krk", target_column="class", task=TaskType.CLASSIFICATION
    ),
    "lahman_2014": FITDatasetDefaults(
        target_table="salaries", target_column="salary", task=TaskType.REGRESSION
    ),
    "legalActs": FITDatasetDefaults(
        target_table="legalacts",
        target_column="ActKind",
        task=TaskType.CLASSIFICATION,
        db_distinct_counter="fetchall_unidecode_strip_ci",
    ),
    "medical": FITDatasetDefaults(
        target_table="Examination",
        target_column="Thrombosis",
        task=TaskType.CLASSIFICATION,
    ),
    "Mesh": FITDatasetDefaults(
        target_table="mesh", target_column="num", task=TaskType.REGRESSION
    ),
    "Mondial": FITDatasetDefaults(
        target_table="target", target_column="Target", task=TaskType.CLASSIFICATION
    ),
    "Mooney_Family": FITDatasetDefaults(
        target_table="uncle", target_column="?", task=TaskType.CLASSIFICATION
    ),
    "MuskSmall": FITDatasetDefaults(
        target_table="molecule", target_column="class", task=TaskType.CLASSIFICATION
    ),
    "mutagenesis": FITDatasetDefaults(
        target_table="molecule", target_column="mutagenic", task=TaskType.CLASSIFICATION
    ),
    "nations": FITDatasetDefaults(
        target_table="stat", target_column="femaleworkers", task=TaskType.CLASSIFICATION
    ),
    "NBA": FITDatasetDefaults(
        target_table="Game", target_column="ResultOfTeam1", task=TaskType.CLASSIFICATION
    ),
    "NCAA": FITDatasetDefaults(
        target_table="target", target_column="team_id1_wins", task=TaskType.CLASSIFICATION
    ),
    "northwind": FITDatasetDefaults(
        target_table="Orders", target_column="Freight", task=TaskType.REGRESSION
    ),
    "Pima": FITDatasetDefaults(
        target_table="pima", target_column="arg2", task=TaskType.CLASSIFICATION
    ),
    "PremierLeague": FITDatasetDefaults(
        target_table="Matches",
        target_column="ResultOfTeamHome",
        task=TaskType.CLASSIFICATION,
    ),
    "PTE": FITDatasetDefaults(
        target_table="pte_active", target_column="is_active", task=TaskType.CLASSIFICATION
    ),
    "PubMed_Diabetes": FITDatasetDefaults(
        target_table="paper", target_column="class_label", task=TaskType.CLASSIFICATION
    ),
    "pubs": FITDatasetDefaults(
        target_table="titles", target_column="ytd_sales", task=TaskType.REGRESSION
    ),
    "Pyrimidine": FITDatasetDefaults(
        target_table="molecule", target_column="activity", task=TaskType.REGRESSION
    ),
    "restbase": FITDatasetDefaults(
        target_table="generalinfo", target_column="review", task=TaskType.REGRESSION
    ),
    "sakila": FITDatasetDefaults(
        target_table="payment", target_column="amount", task=TaskType.REGRESSION
    ),
    "SalesDB": FITDatasetDefaults(
        target_table="Sales", target_column="Quantity", task=TaskType.REGRESSION
    ),
    "Same_gen": FITDatasetDefaults(
        target_table="target", target_column="target", task=TaskType.CLASSIFICATION
    ),
    "SAP": FITDatasetDefaults(
        target_table="Mailings1_2", target_column="RESPONSE", task=TaskType.CLASSIFICATION
    ),
    "SAT": FITDatasetDefaults(
        target_table="fault", target_column="tf", task=TaskType.CLASSIFICATION
    ),
    "Seznam": FITDatasetDefaults(
        target_table="probehnuto", target_column="kc_proklikano", task=TaskType.REGRESSION
    ),
    "SFScores": FITDatasetDefaults(
        target_table="inspections", target_column="score", task=TaskType.REGRESSION
    ),
    "Shakespeare": FITDatasetDefaults(
        target_table="paragraphs",
        target_column="character_id",
        task=TaskType.CLASSIFICATION,
    ),
    "stats": FITDatasetDefaults(
        target_table="users", target_column="Reputation", task=TaskType.REGRESSION
    ),
    "Student_loan": FITDatasetDefaults(
        target_table="no_payment_due", target_column="bool", task=TaskType.CLASSIFICATION
    ),
    "Toxicology": FITDatasetDefaults(
        target_table="molecule", target_column="label", task=TaskType.CLASSIFICATION
    ),
    "tpcc": FITDatasetDefaults(
        target_table="C_Customer", target_column="c_credit", task=TaskType.CLASSIFICATION
    ),
    "tpcd": FITDatasetDefaults(
        target_table="dss_customer",
        target_column="c_mktsegment",
        task=TaskType.CLASSIFICATION,
    ),
    "tpcds": FITDatasetDefaults(
        target_table="customer",
        target_column="c_preferred_cust_flag",
        task=TaskType.CLASSIFICATION,
    ),
    "tpch": FITDatasetDefaults(
        target_table="customer", target_column="c_acctbal", task=TaskType.REGRESSION
    ),
    "trains": FITDatasetDefaults(
        target_table="trains", target_column="direction", task=TaskType.CLASSIFICATION
    ),
    "Triazine": FITDatasetDefaults(
        target_table="molecule", target_column="activity", task=TaskType.REGRESSION
    ),
    "university": FITDatasetDefaults(
        target_table="student", target_column="intelligence", task=TaskType.CLASSIFICATION
    ),
    "UTube": FITDatasetDefaults(
        target_table="utube_states", target_column="class", task=TaskType.CLASSIFICATION
    ),
    "UW_std": FITDatasetDefaults(
        target_table="person", target_column="inPhase", task=TaskType.CLASSIFICATION
    ),
    "VisualGenome": FITDatasetDefaults(
        target_table="IMG_OBJ", target_column="OBJ_CLASS_ID", task=TaskType.CLASSIFICATION
    ),
    "voc": FITDatasetDefaults(
        target_table="voyages",
        target_column="arrival_harbour",
        task=TaskType.CLASSIFICATION,
    ),
    "Walmart": FITDatasetDefaults(
        target_table="train", target_column="units", task=TaskType.REGRESSION
    ),
    "WebKP": FITDatasetDefaults(
        target_table="webpage", target_column="class_label", task=TaskType.CLASSIFICATION
    ),
    "world": FITDatasetDefaults(
        target_table="Country", target_column="Continent", task=TaskType.CLASSIFICATION
    ),
}
