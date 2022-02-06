# Copyright 2014-2020 Chris Cummins <chrisc.101@gmail.com>.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for //labm8/py:sqlutil."""
import os
import pathlib
import typing

import sqlalchemy as sql
from labm8.py import sqlutil, test
from labm8.py.test_data import test_protos_pb2
from sqlalchemy.ext import declarative

FLAGS = test.FLAGS


def test_CreateEngine_sqlite_not_found(tempdir: pathlib.Path):
    """Test DatabaseNotFound for non-existent SQLite database."""
    with test.Raises(sqlutil.DatabaseNotFound) as e_ctx:
        sqlutil.CreateEngine(
            f"sqlite:///{tempdir.absolute()}/db.db",
            must_exist=True,
        )
    assert e_ctx.value.url == f"sqlite:///{tempdir.absolute()}/db.db"
    assert str(e_ctx.value) == (
        f"Database not found: " f"'sqlite:///{tempdir.absolute()}/db.db'"
    )


def test_CreateEngine_sqlite_invalid_relpath():
    """Test that relative paths are disabled."""
    with test.Raises(ValueError) as e_ctx:
        sqlutil.CreateEngine(f"sqlite:///relative.db")
    assert str(e_ctx.value) == "Relative path to SQLite database is not allowed"


def test_CreateEngine_error_if_sqlite_in_memory_must_exist():
    """Error is raised if in-memory "must exist" database requested."""
    with test.Raises(ValueError) as e_ctx:
        sqlutil.CreateEngine("sqlite://", must_exist=True)
    assert str(e_ctx.value) == (
        "must_exist=True not valid for in-memory " "SQLite database"
    )


def test_CreateEngine_sqlite_created(tempdir: pathlib.Path):
    """Test that SQLite database is found."""
    sqlutil.CreateEngine(f"sqlite:///{tempdir}/db.db")
    assert (tempdir / "db.db").is_file()


def test_CreateEngine_sqlite_from_file(tempdir: pathlib.Path):
    """Test file:// sqlite URL."""
    db_path = tempdir / "sqlite.db"
    with open(tempdir / "sqlite.txt", "w") as f:
        f.write(f"sqlite:///{db_path}")
    sqlutil.CreateEngine(f"file://{tempdir}/sqlite.txt")
    assert db_path.is_file()


def test_CreateEngine_sqlite_from_file_with_suffix(tempdir: pathlib.Path):
    """Test file:// sqlite URL with suffix."""
    db_path = tempdir / "sqlite.db"
    with open(tempdir / "sqlite.txt", "w") as f:
        f.write(f"sqlite:///{tempdir}")
    sqlutil.CreateEngine(f"file://{tempdir}/sqlite.txt?/sqlite.db")
    assert db_path.is_file()


def test_ResolveUrl_unmodified():
    assert sqlutil.ResolveUrl("sqlite:///tmp/foo.db") == "sqlite:///tmp/foo.db"


def test_ResolveUrl_unexpanded_shell_variable():
    if "FOO" in os.environ:
        del os.environ["FOO"]
    assert sqlutil.ResolveUrl("sqlite:///tmp/$FOO.db") == "sqlite:///tmp/$FOO.db"
    assert sqlutil.ResolveUrl("sqlite:///tmp/${FOO}.db") == "sqlite:///tmp/${FOO}.db"


def test_ResolveUrl_expanded_shell_variable():
    os.environ["FOO"] = "abc"
    assert sqlutil.ResolveUrl("sqlite:///tmp/$FOO.db") == "sqlite:///tmp/abc.db"
    assert sqlutil.ResolveUrl("sqlite:///tmp/${FOO}.db") == "sqlite:///tmp/abc.db"


def test_ResolveUrl_path_not_found(tempdir: pathlib.Path):
    with test.Raises(FileNotFoundError):
        sqlutil.ResolveUrl(f"file:///{tempdir}/file.txt")


def test_ResolveUrl_path_is_directory(tempdir: pathlib.Path):
    with test.Raises(FileNotFoundError):
        sqlutil.ResolveUrl(f"file:///{tempdir}")


def test_ResolveUrl_mysql_charset():
    FLAGS.mysql_assume_utf8_charset = True
    assert sqlutil.ResolveUrl("mysql://foo/bar") == "mysql://foo/bar?charset=utf8"


def test_ResolveUrl_no_mysql_charset():
    FLAGS.mysql_assume_utf8_charset = False
    assert sqlutil.ResolveUrl("mysql://foo/bar") == "mysql://foo/bar"


def test_ResolveUrl_mysql_charset_with_file(tempdir: pathlib.Path):
    FLAGS.mysql_assume_utf8_charset = True
    with open(tempdir / "mysql.txt", "w") as f:
        f.write(f"mysql://foo/bar")
    assert (
        sqlutil.ResolveUrl(f"file://{tempdir}/mysql.txt")
        == "mysql://foo/bar?charset=utf8"
    )


def test_AllColumnNames_two_fields():
    """Test that column names are returned."""
    base = declarative.declarative_base()

    class Table(base, sqlutil.TablenameFromClassNameMixin):
        """A table containing two columns."""

        col_a = sql.Column(sql.Integer, primary_key=True)
        col_b = sql.Column(sql.Integer)

    assert sqlutil.ColumnNames(Table) == ["col_a", "col_b"]


def test_AllColumnNames_two_fields_model_instance():
    """Test that column names are returned on instance."""
    base = declarative.declarative_base()

    class Table(base, sqlutil.TablenameFromClassNameMixin):
        """A table containing two columns."""

        col_a = sql.Column(sql.Integer, primary_key=True)
        col_b = sql.Column(sql.Integer)

    instance = Table(col_a=1, col_b=2)
    assert sqlutil.ColumnNames(instance) == ["col_a", "col_b"]


def test_AllColumnNames_invalid_object():
    """TypeError raised when called on an invalid object."""

    class NotAModel(object):
        col_a = sql.Column(sql.Integer, primary_key=True)

    with test.Raises(TypeError):
        sqlutil.ColumnNames(NotAModel)


def test_Session_re_use():
    base = declarative.declarative_base()

    class Table(base, sqlutil.TablenameFromClassNameMixin):
        """A table containing a single 'value' primary key."""

        value = sql.Column(sql.Integer, primary_key=True)

    # Create and populate a test database.
    db = sqlutil.Database("sqlite://", base)
    with db.Session(commit=True) as s:
        s.GetOrAdd(Table, value=42)

    # Test Session pass-thru.
    with db.Session() as session:
        with db.Session(session) as session:
            assert session.query(Table).count() == 1

    # Test Session with default constructed session.
    with db.Session(session=None) as session:
        assert session.query(Table).count() == 1


def test_Session_GetOrAdd():
    """Test that GetOrAdd() does not create duplicates."""
    base = declarative.declarative_base()

    class Table(base, sqlutil.TablenameFromClassNameMixin):
        """A table containing a single 'value' primary key."""

        value = sql.Column(sql.Integer, primary_key=True)

    # Create the database.
    db = sqlutil.Database("sqlite://", base)

    # Create an entry in the database.
    with db.Session(commit=True) as s:
        s.GetOrAdd(Table, value=42)

    # Check that the entry has been added.
    with db.Session() as s:
        assert s.query(Table).count() == 1
        assert s.query(Table).one().value == 42

    # Create another entry. Since a row already exists in the database, this
    # doesn't add anything.
    with db.Session(commit=True) as s:
        s.GetOrAdd(Table, value=42)

    # Check that the database is unchanged.
    with db.Session() as s:
        assert s.query(Table).count() == 1
        assert s.query(Table).one().value == 42


def test_Close_raises_error_on_session():
    """Test that session creation fails after call to Database.Close()."""
    base = declarative.declarative_base()

    class Table(base, sqlutil.TablenameFromClassNameMixin):
        """A table containing a single 'value' primary key."""

        value = sql.Column(sql.Integer, primary_key=True)

    # Create the database.
    db = sqlutil.Database("sqlite://", base)
    db.Close()
    with test.Raises(sql.exc.OperationalError):
        # Try and perform a query on the closed database.
        with db.Session(commit=True) as s:
            s.query(Table).all()


class AbstractTestMessage(
    sqlutil.ProtoBackedMixin,
    sqlutil.TablenameFromClassNameMixin,
):
    """A table containing a single 'value' primary key."""

    proto_t = test_protos_pb2.TestMessage

    string = sql.Column(sql.String, primary_key=True)
    number = sql.Column(sql.Integer)

    def SetProto(self, proto: test_protos_pb2.TestMessage) -> None:
        """Set a protocol buffer representation."""
        proto.string = self.string
        proto.number = self.number

    @staticmethod
    def FromProto(proto) -> typing.Dict[str, typing.Any]:
        """Instantiate an object from protocol buffer message."""
        return {
            "string": proto.string,
            "number": proto.number,
        }


def test_ProtoBackedMixin_FromProto():
    """Test FromProto constructor for proto backed tables."""
    base = declarative.declarative_base()

    class TestMessage(AbstractTestMessage, base):
        pass

    proto = test_protos_pb2.TestMessage(string="Hello, world!", number=42)
    row = TestMessage(**TestMessage.FromProto(proto))
    assert row.string == "Hello, world!"
    assert row.number == 42


def test_ProtoBackedMixin_SetProto():
    """Test SetProto method for proto backed tables."""
    base = declarative.declarative_base()

    class TestMessage(AbstractTestMessage, base):
        pass

    proto = test_protos_pb2.TestMessage()
    TestMessage(string="Hello, world!", number=42).SetProto(proto)
    assert proto.string == "Hello, world!"
    assert proto.number == 42


def test_ProtoBackedMixin_ToProto():
    """Test FromProto constructor for proto backed tables."""
    base = declarative.declarative_base()

    class TestMessage(AbstractTestMessage, base):
        pass

    row = TestMessage(string="Hello, world!", number=42)
    proto = row.ToProto()
    assert proto.string == "Hello, world!"
    assert proto.number == 42


def test_ProtoBackedMixin_FromFile(tempdir: pathlib.Path):
    """Test FromProto constructor for proto backed tables."""
    base = declarative.declarative_base()

    class TestMessage(AbstractTestMessage, base):
        pass

    with open(tempdir / "proto.pbtxt", "w") as f:
        f.write('string: "Hello, world!"')

    row = TestMessage(**TestMessage.FromFile(tempdir / "proto.pbtxt"))
    assert row.string == "Hello, world!"


def test_ColumnTypes_BinaryArray():
    """Test that column type can be instantiated."""
    base = declarative.declarative_base()

    class Table(base):
        __tablename__ = "test"
        primary_key = sql.Column(sql.Integer, primary_key=True)
        col = sql.Column(sqlutil.ColumnTypes.BinaryArray(16))

    db = sqlutil.Database("sqlite://", base)
    with db.Session(commit=True) as s:
        s.add(Table(col="abc".encode("utf-8")))


def test_ColumnTypes_UnboundedUnicodeText():
    """Test that column type can be instantiated."""
    base = declarative.declarative_base()

    class Table(base):
        __tablename__ = "test"
        primary_key = sql.Column(sql.Integer, primary_key=True)
        col = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText())

    db = sqlutil.Database("sqlite://", base)
    with db.Session(commit=True) as s:
        s.add(Table(col="abc"))


def test_Random_order_by():

    base = declarative.declarative_base()

    class Table(base):
        __tablename__ = "test"
        col = sql.Column(sql.Integer, primary_key=1)

    db = sqlutil.Database("sqlite://", base)
    with db.Session() as s:
        s.add(Table(col=1))
        s.add(Table(col=2))
        s.add(Table(col=3))
        s.flush()

        random_row = s.query(Table).order_by(db.Random()).first()
        assert random_row.col in {1, 2, 3}


def test_ResilientAddManyAndCommit_no_conflicts():
    base = declarative.declarative_base()

    class Table(base):
        __tablename__ = "test"
        col = sql.Column(sql.Integer, primary_key=1)

    db = sqlutil.Database("sqlite://", base)
    mapped = [Table(col=1), Table(col=2), Table(col=3)]
    assert not sqlutil.ResilientAddManyAndCommit(db, mapped)

    with db.Session() as s:
        assert s.query(Table).count() == 3


def test_ResilientAddManyAndCommit_conflicting_primary_key():
    base = declarative.declarative_base()

    class Table(base):
        __tablename__ = "test"
        col = sql.Column(sql.Integer, primary_key=1)

    db = sqlutil.Database("sqlite://", base)

    # Adding objects with conflicting primary keys will raise an error.
    # In which case, one will be committed succesfully. The other will be
    # returned.
    mapped = [
        Table(col=1),
        Table(col=1),
        Table(col=1),
        Table(col=1),
        Table(col=1),
    ]
    failures = sqlutil.ResilientAddManyAndCommit(db, mapped)

    with db.Session() as s:
        assert s.query(Table).count() == 1

    assert len(failures) == 4
    assert failures[0].col == 1
    assert failures[1].col == 1
    assert failures[2].col == 1
    assert failures[3].col == 1


def test_QueryToString_inline_literal_int():
    base = declarative.declarative_base()

    class Table(base):
        __tablename__ = "test"
        col = sql.Column(sql.Integer, primary_key=1)

    db = sqlutil.Database("sqlite://", base)
    with db.Session() as session:
        query = session.query(Table.col).filter(Table.col == 5)

    assert sqlutil.QueryToString(query) == (
        "SELECT test.col \nFROM test \nWHERE test.col = 5"
    )


Base = declarative.declarative_base()


class Table(Base):
    __tablename__ = "test"
    col = sql.Column(sql.Integer, primary_key=1)


@test.Fixture(scope="function")
def db(tempdir: pathlib.Path) -> Table:
    yield sqlutil.Database(f"sqlite:///{tempdir}/db", Base)


@test.Parametrize("max_buffer_size", (1, 1024))
@test.Parametrize("max_buffer_length", (1, 5))
@test.Parametrize("max_seconds_since_flush", (1, 10))
def test_BufferedDatabaseWriter_add_one(
    db: sqlutil.Database,
    max_buffer_size: int,
    max_buffer_length: int,
    max_seconds_since_flush: float,
):
    with sqlutil.BufferedDatabaseWriter(
        db,
        max_buffer_length=max_buffer_length,
        max_buffer_size=max_buffer_size,
        max_seconds_since_flush=max_seconds_since_flush,
    ) as writer:
        writer.AddOne(Table(col=1))
        writer.AddOne(Table(col=2))
        writer.AddOne(Table(col=3))

    with db.Session() as s:
        assert s.query(Table).count() == 3


@test.Parametrize("max_buffer_size", (1, 1024))
@test.Parametrize("max_buffer_length", (1, 5))
@test.Parametrize("max_seconds_since_flush", (1, 10))
def test_BufferedDatabaseWriter_add_many(
    db: sqlutil.Database,
    max_buffer_size: int,
    max_buffer_length: int,
    max_seconds_since_flush: float,
):
    with sqlutil.BufferedDatabaseWriter(
        db,
        max_buffer_length=max_buffer_length,
        max_buffer_size=max_buffer_size,
        max_seconds_since_flush=max_seconds_since_flush,
    ) as writer:
        writer.AddMany([Table(col=1), Table(col=2), Table(col=3)])

    with db.Session() as s:
        assert s.query(Table).count() == 3


@test.Parametrize("max_buffer_size", (1, 1024))
@test.Parametrize("max_buffer_length", (1, 5))
@test.Parametrize("max_seconds_since_flush", (1, 10))
def test_BufferedDatabaseWriter_lambda_op(
    db: sqlutil.Database,
    max_buffer_size: int,
    max_buffer_length: int,
    max_seconds_since_flush: float,
):
    """Test buffered writer when there is a lambda op in the buffer."""
    with sqlutil.BufferedDatabaseWriter(
        db,
        max_buffer_length=max_buffer_length,
        max_buffer_size=max_buffer_size,
        max_seconds_since_flush=max_seconds_since_flush,
    ) as writer:
        writer.AddOne(Table(col=1))
        writer.AddLambdaOp(lambda session: None)
        writer.AddOne(Table(col=2))

    with db.Session() as session:
        assert session.query(Table).count() == 2


@test.Parametrize("max_buffer_size", (1, 1024))
@test.Parametrize("max_buffer_length", (1, 5))
@test.Parametrize("max_seconds_since_flush", (1, 10))
def test_BufferedDatabaseWriter_lambda_op_deletes_previous_entry(
    db: sqlutil.Database,
    max_buffer_size: int,
    max_buffer_length: int,
    max_seconds_since_flush: float,
):
    """Test buffered writer when there is a lambda op that deletes entries that
    still buffered, and haven't yet been committed to the database."""
    with sqlutil.BufferedDatabaseWriter(
        db,
        max_buffer_length=max_buffer_length,
        max_buffer_size=max_buffer_size,
        max_seconds_since_flush=max_seconds_since_flush,
    ) as writer:
        writer.AddMany([Table(col=i) for i in range(64)])
        # This op deletes all of the table entries added above.
        writer.AddLambdaOp(
            lambda session: session.query(Table).delete(synchronize_session=False)
        )
        writer.AddOne(Table(col=2))

    with db.Session() as session:
        assert session.query(Table).count() == 1


def test_PluralTablenameFromCamelCapsClassNameMixin():
    base = declarative.declarative_base()

    class FooBar(base, sqlutil.PluralTablenameFromCamelCapsClassNameMixin):
        col = sql.Column(sql.Integer, primary_key=1)

    assert FooBar.__tablename__ == "foo_bars"


def test_EnableSqliteForeignKeysCallback_delete_cascade():
    """Test that parent can't be deleted with foreign key constraints set."""
    FLAGS.sqlite_enable_foreign_keys = True

    base = declarative.declarative_base()

    class Parent(base, sqlutil.TablenameFromClassNameMixin):
        id = sql.Column(sql.Integer, primary_key=True)
        child = sql.orm.relationship(
            "Child", uselist=False, cascade="all, delete-orphan"
        )

    class Child(base, sqlutil.TablenameFromClassNameMixin):
        id = sql.Column(sql.Integer, sql.ForeignKey("parent.id"), primary_key=True)

    db = sqlutil.Database("sqlite://", base)
    with db.Session() as session:
        session.add(Parent(id=1, child=Child(id=1)))
        session.commit()

        with test.Raises(sql.exc.IntegrityError):
            session.query(Parent).filter(Parent.id == 1).delete()


def test_EnableSqliteForeignKeysCallback_delete_cascade_without_foreign_key_constraints():
    """Test that parent can't be deleted with foreign key constraints set."""
    FLAGS.sqlite_enable_foreign_keys = False

    base = declarative.declarative_base()

    class Parent(base, sqlutil.TablenameFromClassNameMixin):
        id = sql.Column(sql.Integer, primary_key=True)
        child = sql.orm.relationship(
            "Child", uselist=False, cascade="all, delete-orphan"
        )

    class Child(base, sqlutil.TablenameFromClassNameMixin):
        id = sql.Column(sql.Integer, sql.ForeignKey("parent.id"), primary_key=True)

    db = sqlutil.Database("sqlite://", base)
    with db.Session() as session:
        session.add(Parent(id=1, child=Child(id=1)))
        session.commit()

        # WARNING: As this test demonstrates this, when
        # --sqlite_enable_foreign_keys=false foreign key constraints are not
        # enforced by SQLite. This database is now in a corrupt state where an
        # orphaned child exists without a parent, be careful of this!!
        session.query(Parent).filter(Parent.id == 1).delete()
        session.commit()


if __name__ == "__main__":
    test.Main()
