from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, Optional, Tuple, Union

from db_transformer.data.dataset_defaults.fit_dataset_fixes import fix_citeseer_schema
from db_transformer.db.distinct_cnt_retrieval import DBDistinctCounter
from db_transformer.db.schema_autodetect import BuiltinDBDistinctCounter
from db_transformer.schema.schema import Schema


class TaskType(Enum):
    CLASSIFICATION = 1
    REGRESSION = 2


@dataclass
class FITDatasetDefaults:
    _target_table: str
    _target_column: str
    task: TaskType
    schema_fixer: Optional[Callable[[Schema], None]] = None
    db_distinct_counter: Union[DBDistinctCounter, BuiltinDBDistinctCounter] = 'db_distinct'
    force_collation: Optional[str] = None
    
    case_sensitive = True
    
    @property
    def target_table(self) -> str:
        return self._target_table if self.case_sensitive else self._target_table.lower()
    
    @property
    def target_column(self) -> str:
        return self._target_column if self.case_sensitive else self._target_column.lower()

    @property
    def target(self) -> Tuple[str, str]:
        return self.target_table, self.target_column


FIT_DATASET_DEFAULTS: Dict[str, FITDatasetDefaults] = {
    'Accidents': FITDatasetDefaults(_target_table='nesreca', _target_column='klas_nesreca', task=TaskType.CLASSIFICATION),
    'AdventureWorks2014': FITDatasetDefaults(_target_table='SalesOrderHeader', _target_column='TotalDue', task=TaskType.REGRESSION),
    'Airline': FITDatasetDefaults(_target_table='On_Time_On_Time_Performance_2016_1', _target_column='ArrDel15', task=TaskType.CLASSIFICATION),
    'Atherosclerosis': FITDatasetDefaults(_target_table='Death', _target_column='PRICUMR', task=TaskType.CLASSIFICATION),
    'Basketball_men': FITDatasetDefaults(_target_table='teams', _target_column='rank', task=TaskType.REGRESSION),
    'Basketball_women': FITDatasetDefaults(_target_table='teams', _target_column='playoff', task=TaskType.CLASSIFICATION),
    'Biodegradability': FITDatasetDefaults(_target_table='molecule', _target_column='activity', task=TaskType.REGRESSION),
    'Bupa': FITDatasetDefaults(_target_table='bupa', _target_column='arg2', task=TaskType.CLASSIFICATION),
    'Carcinogenesis': FITDatasetDefaults(_target_table='canc', _target_column='class', task=TaskType.CLASSIFICATION),
    'ccs': FITDatasetDefaults(_target_table='transactions_1k', _target_column='Price', task=TaskType.REGRESSION),
    'CDESchools': FITDatasetDefaults(_target_table='satscores', _target_column='PctGE1500', task=TaskType.REGRESSION),
    'Chess': FITDatasetDefaults(_target_table='game', _target_column='game_result', task=TaskType.CLASSIFICATION),
    'CiteSeer': FITDatasetDefaults(_target_table='paper', _target_column='class_label', task=TaskType.CLASSIFICATION,
                                   schema_fixer=fix_citeseer_schema),
    'classicmodels': FITDatasetDefaults(_target_table='payments', _target_column='amount', task=TaskType.REGRESSION),
    'ConsumerExpenditures': FITDatasetDefaults(_target_table='EXPENDITURES', _target_column='GIFT', task=TaskType.CLASSIFICATION),
    'CORA': FITDatasetDefaults(_target_table='paper', _target_column='class_label', task=TaskType.CLASSIFICATION),
    'Countries': FITDatasetDefaults(_target_table='target', _target_column='2012', task=TaskType.REGRESSION),
    'CraftBeer': FITDatasetDefaults(_target_table='breweries', _target_column='state', task=TaskType.CLASSIFICATION),
    'Credit': FITDatasetDefaults(_target_table='member', _target_column='region_no', task=TaskType.CLASSIFICATION),
    'cs': FITDatasetDefaults(_target_table='target_churn', _target_column='target_churn', task=TaskType.CLASSIFICATION),
    'Dallas': FITDatasetDefaults(_target_table='incidents', _target_column='subject_statuses', task=TaskType.CLASSIFICATION),
    'DCG': FITDatasetDefaults(_target_table='sentences', _target_column='class', task=TaskType.CLASSIFICATION),
    'Dunur': FITDatasetDefaults(_target_table='target', _target_column='is_dunur', task=TaskType.CLASSIFICATION),
    'Elti': FITDatasetDefaults(_target_table='target', _target_column='is_elti', task=TaskType.CLASSIFICATION),
    'employee': FITDatasetDefaults(_target_table='salaries', _target_column='salary', task=TaskType.REGRESSION),
    'ErgastF1': FITDatasetDefaults(_target_table='target', _target_column='win', task=TaskType.CLASSIFICATION),
    'Facebook': FITDatasetDefaults(_target_table='feat', _target_column='gender1', task=TaskType.CLASSIFICATION),
    'financial': FITDatasetDefaults(_target_table='loan', _target_column='status', task=TaskType.CLASSIFICATION),
    'FNHK': FITDatasetDefaults(_target_table='pripady', _target_column='Delka_hospitalizace', task=TaskType.REGRESSION),
    'ftp': FITDatasetDefaults(_target_table='session', _target_column='gender', task=TaskType.CLASSIFICATION),
    'geneea': FITDatasetDefaults(_target_table='hl_hlasovani', _target_column='vysledek', task=TaskType.CLASSIFICATION,
                                 force_collation='utf8mb3_unicode_ci'),
    'genes': FITDatasetDefaults(_target_table='Classification', _target_column='Localization', task=TaskType.CLASSIFICATION),
    'GOSales': FITDatasetDefaults(_target_table='go_1k', _target_column='Quantity', task=TaskType.REGRESSION),
    'Grants': FITDatasetDefaults(_target_table='awards', _target_column='award_amount', task=TaskType.REGRESSION),
    'Hepatitis_std': FITDatasetDefaults(_target_table='dispat', _target_column='Type', task=TaskType.CLASSIFICATION),
    'Hockey': FITDatasetDefaults(_target_table='Master', _target_column='shootCatch', task=TaskType.CLASSIFICATION),
    'imdb_ijs': FITDatasetDefaults(_target_table='actors', _target_column='gender', task=TaskType.CLASSIFICATION,
                                   db_distinct_counter='fetchall_unidecode_strip_ci'),
    'imdb_MovieLens': FITDatasetDefaults(_target_table='users', _target_column='u_gender', task=TaskType.CLASSIFICATION),
    'KRK': FITDatasetDefaults(_target_table='krk', _target_column='class', task=TaskType.CLASSIFICATION),
    'lahman_2014': FITDatasetDefaults(_target_table='salaries', _target_column='salary', task=TaskType.REGRESSION),
    'legalActs': FITDatasetDefaults(_target_table='legalacts', _target_column='ActKind', task=TaskType.CLASSIFICATION,
                                    db_distinct_counter='fetchall_unidecode_strip_ci'),
    'medical': FITDatasetDefaults(_target_table='Examination', _target_column='Thrombosis', task=TaskType.CLASSIFICATION),
    'Mesh': FITDatasetDefaults(_target_table='mesh', _target_column='num', task=TaskType.REGRESSION),
    'Mondial': FITDatasetDefaults(_target_table='target', _target_column='Target', task=TaskType.CLASSIFICATION),
    'Mooney_Family': FITDatasetDefaults(_target_table='uncle', _target_column='?', task=TaskType.CLASSIFICATION),
    'MuskSmall': FITDatasetDefaults(_target_table='molecule', _target_column='class', task=TaskType.CLASSIFICATION),
    'mutagenesis': FITDatasetDefaults(_target_table='molecule', _target_column='mutagenic', task=TaskType.CLASSIFICATION),
    'nations': FITDatasetDefaults(_target_table='stat', _target_column='femaleworkers', task=TaskType.CLASSIFICATION),
    'NBA': FITDatasetDefaults(_target_table='Game', _target_column='ResultOfTeam1', task=TaskType.CLASSIFICATION),
    'NCAA': FITDatasetDefaults(_target_table='target', _target_column='team_id1_wins', task=TaskType.CLASSIFICATION),
    'northwind': FITDatasetDefaults(_target_table='Orders', _target_column='Freight', task=TaskType.REGRESSION),
    'Pima': FITDatasetDefaults(_target_table='pima', _target_column='arg2', task=TaskType.CLASSIFICATION),
    'PremierLeague': FITDatasetDefaults(_target_table='Matches', _target_column='ResultOfTeamHome', task=TaskType.CLASSIFICATION),
    'PTE': FITDatasetDefaults(_target_table='pte_active', _target_column='is_active', task=TaskType.CLASSIFICATION),
    'PubMed_Diabetes': FITDatasetDefaults(_target_table='paper', _target_column='class_label', task=TaskType.CLASSIFICATION),
    'pubs': FITDatasetDefaults(_target_table='titles', _target_column='ytd_sales', task=TaskType.REGRESSION),
    'Pyrimidine': FITDatasetDefaults(_target_table='molecule', _target_column='activity', task=TaskType.REGRESSION),
    'restbase': FITDatasetDefaults(_target_table='generalinfo', _target_column='review', task=TaskType.REGRESSION),
    'sakila': FITDatasetDefaults(_target_table='payment', _target_column='amount', task=TaskType.REGRESSION),
    'SalesDB': FITDatasetDefaults(_target_table='Sales', _target_column='Quantity', task=TaskType.REGRESSION),
    'Same_gen': FITDatasetDefaults(_target_table='target', _target_column='target', task=TaskType.CLASSIFICATION),
    'SAP': FITDatasetDefaults(_target_table='Mailings1_2', _target_column='RESPONSE', task=TaskType.CLASSIFICATION),
    'SAT': FITDatasetDefaults(_target_table='fault', _target_column='tf', task=TaskType.CLASSIFICATION),
    'Seznam': FITDatasetDefaults(_target_table='probehnuto', _target_column='kc_proklikano', task=TaskType.REGRESSION),
    'SFScores': FITDatasetDefaults(_target_table='inspections', _target_column='score', task=TaskType.REGRESSION),
    'Shakespeare': FITDatasetDefaults(_target_table='paragraphs', _target_column='character_id', task=TaskType.CLASSIFICATION),
    'stats': FITDatasetDefaults(_target_table='users', _target_column='Reputation', task=TaskType.REGRESSION),
    'Student_loan': FITDatasetDefaults(_target_table='no_payment_due', _target_column='bool', task=TaskType.CLASSIFICATION),
    'Toxicology': FITDatasetDefaults(_target_table='molecule', _target_column='label', task=TaskType.CLASSIFICATION),
    'tpcc': FITDatasetDefaults(_target_table='C_Customer', _target_column='c_credit', task=TaskType.CLASSIFICATION),
    'tpcd': FITDatasetDefaults(_target_table='dss_customer', _target_column='c_mktsegment', task=TaskType.CLASSIFICATION),
    'tpcds': FITDatasetDefaults(_target_table='customer', _target_column='c_preferred_cust_flag', task=TaskType.CLASSIFICATION),
    'tpch': FITDatasetDefaults(_target_table='customer', _target_column='c_acctbal', task=TaskType.REGRESSION),
    'trains': FITDatasetDefaults(_target_table='trains', _target_column='direction', task=TaskType.CLASSIFICATION),
    'Triazine': FITDatasetDefaults(_target_table='molecule', _target_column='activity', task=TaskType.REGRESSION),
    'university': FITDatasetDefaults(_target_table='student', _target_column='intelligence', task=TaskType.CLASSIFICATION),
    'UTube': FITDatasetDefaults(_target_table='utube_states', _target_column='class', task=TaskType.CLASSIFICATION),
    'UW_std': FITDatasetDefaults(_target_table='person', _target_column='inPhase', task=TaskType.CLASSIFICATION),
    'VisualGenome': FITDatasetDefaults(_target_table='IMG_OBJ', _target_column='OBJ_CLASS_ID', task=TaskType.CLASSIFICATION),
    'voc': FITDatasetDefaults(_target_table='voyages', _target_column='arrival_harbour', task=TaskType.CLASSIFICATION),
    'Walmart': FITDatasetDefaults(_target_table='train', _target_column='units', task=TaskType.REGRESSION),
    'WebKP': FITDatasetDefaults(_target_table='webpage', _target_column='class_label', task=TaskType.CLASSIFICATION),
    'world': FITDatasetDefaults(_target_table='Country', _target_column='Continent', task=TaskType.CLASSIFICATION)
}
