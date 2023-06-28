from typing import List

from sqlalchemy import create_engine, Table, MetaData, Column, select, func, table
from sqlalchemy.orm import Session


def copy_database(src_connection_str, dst_connection_str, tables: List[str]):
    src_engine = create_engine(src_connection_str)
    dst_engine = create_engine(dst_connection_str)

    with dst_engine.connect() as dst_connection, Session(src_engine) as src_session:
        dst_metadata = MetaData()
        dst_metadata.reflect(bind=dst_engine)

        src_metadata = MetaData()
        src_metadata.reflect(bind=src_engine)

        create_tables = []
        for table_name in tables:
            src_table = Table(table_name, src_metadata)

            columns = [
                Column(column.name, column.type)
                for column in src_table.columns
            ]

            create_tables.append(Table(table_name, dst_metadata, *columns))

        dst_metadata.create_all(dst_engine, tables=create_tables)

        for table_name, dst_table in zip(tables, create_tables):
            # TODO: Insert/Select in batch
            for res in src_session.execute(select("*", table(table_name))).all():
                dst_connection.execute(dst_table.insert().values(res))


def get_table_len(table_name: str, session: Session) -> int:
    return int(session.query(func.count(None)).select_from(table(table_name)).scalar())
