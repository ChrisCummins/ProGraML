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
import sqlalchemy as sql
from labm8.py import pdutil, sqlutil
from sqlalchemy.ext import declarative


def test_QueryToDataFrame_column_names():
    """Test that expected column names are set."""
    base = declarative.declarative_base()

    class Table(base):
        __tablename__ = "test"
        col_a = sql.Column(sql.Integer, primary_key=True)
        col_b = sql.Column(sql.Integer)

    db = sqlutil.Database("sqlite://", base)
    with db.Session() as s:
        df = pdutil.QueryToDataFrame(s, s.query(Table.col_a, Table.col_b))

    assert list(df.columns.values) == ["col_a", "col_b"]


def test_ModelToDataFrame_column_names():
    """Test that expected column names are set."""
    base = declarative.declarative_base()

    class Table(base):
        __tablename__ = "test"
        col_a = sql.Column(sql.Integer, primary_key=True)
        col_b = sql.Column(sql.Integer)

    db = sqlutil.Database("sqlite://", base)
    with db.Session() as s:
        df = pdutil.ModelToDataFrame(s, Table)

    assert list(df.columns.values) == ["col_a", "col_b"]


def test_QueryToDataFrame_explicit_column_names():
    """Test that expected column names are set."""
    base = declarative.declarative_base()

    class Table(base):
        __tablename__ = "test"
        col_a = sql.Column(sql.Integer, primary_key=True)
        col_b = sql.Column(sql.Integer)

    db = sqlutil.Database("sqlite://", base)
    with db.Session() as s:
        df = pdutil.ModelToDataFrame(s, Table, ["col_b"])

    assert list(df.columns.values) == ["col_b"]
