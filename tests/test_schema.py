import pytest

from db_transformer.data_v2.schema import CategoricalColumnDef, ColumnDef, ColumnType, ForeignKeyColumnDef, NumericColumnDef, Schema, TableSchema, to_obj


class TestColumnDef:
    def test_error_base_class_constructor(self):
        with pytest.raises(TypeError):
            ColumnDef(type=ColumnType.FOREIGN_KEY)

    def test_error_wrong_explicit_type(self):
        with pytest.raises(ValueError):
            NumericColumnDef(type=ColumnType.CATEGORICAL)  # type: ignore

    def test_error_missing_params(self):
        with pytest.raises(TypeError):
            ForeignKeyColumnDef(table='mytable')  # type: ignore


class TestSchema:
    def test_empty_schema(self):
        assert Schema() == Schema.from_obj(dict())
        assert to_obj(Schema()) == dict()

    lhs = Schema(
        mytable=TableSchema(
            mycat=CategoricalColumnDef(card=3),
            mynum=NumericColumnDef(key=True),
            myforeign=ForeignKeyColumnDef(table='mytable2', column='col2')
        ),
        mytable2=TableSchema(
        )
    )

    rhs = Schema.from_obj(dict(
        mytable=dict(
            mycat=dict(type=ColumnType.CATEGORICAL.value, card=3),
            mynum=dict(type=ColumnType.NUMERIC.value, key=True),
            myforeign=dict(type=ColumnType.FOREIGN_KEY.value, table='mytable2', column='col2')
        ),
        mytable2=dict()
    ))

    def test_full_schema_from_obj(self):
        assert self.lhs == self.rhs

    def test_full_schema_to_obj(self):
        assert Schema.from_obj(to_obj(self.lhs)) == self.rhs
        assert to_obj(self.lhs) == to_obj(self.rhs)

    def test_full_schema_not_equal(self):
        assert self.lhs != Schema()
        assert self.lhs.mytable != TableSchema()
        assert self.lhs.mytable.myforeign != ForeignKeyColumnDef(table='NOTmytable2', column='col2')

    def test_full_schema_from_obj_isinstance(self):
        assert isinstance(self.rhs, Schema)
        assert isinstance(self.rhs.mytable, TableSchema)
        assert isinstance(self.rhs.mytable.mycat, CategoricalColumnDef)
        assert isinstance(self.rhs.mytable.mynum, NumericColumnDef)
        assert isinstance(self.rhs.mytable.myforeign, ForeignKeyColumnDef)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main(["-v", __file__]))
