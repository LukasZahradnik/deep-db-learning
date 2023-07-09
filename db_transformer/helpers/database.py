from typing import List

from sqlalchemy import Connection, Table, MetaData, Column, select, func, table

from db_transformer.db.db_inspector import DBInspector


def copy_database(src_inspector: DBInspector, dst: Connection):
    with dst.begin():  # transaction ends at the end of the `with` block
        dst_metadata = MetaData()
        dst_metadata.reflect(bind=dst.engine)

        src_metadata = MetaData()
        src_metadata.reflect(bind=src_inspector.engine)

        tables = src_inspector.get_tables()

        create_tables: List[Table] = []
        for table_name in tables:
            src_table = Table(table_name, src_metadata)

            columns = [
                Column(column.name, column.type)
                for column in src_table.columns
            ]

            create_tables.append(Table(table_name, dst_metadata, *columns))

        dst_metadata.create_all(dst.engine, tables=create_tables)

        for table_name, dst_table in zip(tables, create_tables):
            # TODO: Insert/Select in batch
            for res in src_inspector.connection.execute(select("*", table(table_name))).all():
                dst.execute(dst_table.insert().values(res))


def get_table_len(table_name: str, connection: Connection) -> int:
    query = table(table_name).select().column(func.count(None))
    out = connection.execute(query).scalar()
    return int(out)
