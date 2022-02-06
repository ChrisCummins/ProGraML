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
"""Unit tests for //labm8/py:fmt."""
import pytest
from labm8.py import app, fmt, test

FLAGS = app.FLAGS


def test_IndentList_zero():
    """Indent with 0 spaces is equivalent to the input strings."""
    assert fmt.IndentList(0, ["abc", "d"]) == ["abc", "d"]


def test_IndentList_two():
    """Test indent with two spaces."""
    assert fmt.IndentList(2, ["abc", "d"]) == ["  abc", "  d"]


def test_Indent_zero():
    """Indent with 0 spaces is equivalent to the input string."""
    assert fmt.Indent(0, "abc\nd") == "abc\nd"


def test_Indent_two():
    """Test indent with two spaces."""
    assert fmt.Indent(2, "abc\nd") == "  abc\n  d"


def test_table():
    assert ["foo", "1", "bar", "2"] == fmt.table(
        (("foo", 1), ("bar", 2)),
    ).split()


def test_table_columns():
    assert (["type", "value", "foo", "1", "bar", "2"]) == fmt.table(
        (("foo", 1), ("bar", 2)),
        columns=("type", "value"),
    ).split()


def test_table_bad_columns():
    with test.Raises(fmt.Error):
        fmt.table(
            (("foo", 1), ("bar", 2)),
            columns=("type", "value", "too", "many", "values"),
        )


def test_table_bad_rows():
    with test.Raises(fmt.Error):
        fmt.table((("foo", 1), ("bar", 2), ("car",)))


if __name__ == "__main__":
    test.Main()
