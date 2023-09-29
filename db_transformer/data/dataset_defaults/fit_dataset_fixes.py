from db_transformer.schema.schema import ForeignKeyDef, Schema


def fix_citeseer_schema(schema: Schema):
    schema.cites.foreign_keys += [
        ForeignKeyDef(['cited_paper_id'], 'paper', ['paper_id']),
        ForeignKeyDef(['citing_paper_id'], 'paper', ['paper_id']),
    ]
