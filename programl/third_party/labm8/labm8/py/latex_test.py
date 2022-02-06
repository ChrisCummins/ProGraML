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
"""Unit tests for //labm8/py:latex."""
import pytest
from labm8.py import app, latex, test

FLAGS = app.FLAGS


# escape()
def test_escape():
    assert "foo" == latex.escape("foo")
    assert "123" == latex.escape(123)
    assert "foo\\_bar" == latex.escape("foo_bar")
    assert "foo\\_bar\\_baz" == latex.escape("foo_bar_baz")
    assert "foo\\_\\_bar" == latex.escape("foo__bar")


# write_table_body()
def test_write_table_body():
    assert "1 & foo\\\\\n" "2 & bar\\\\\n" == latex.write_table_body(
        ((1, "foo"), (2, "bar"))
    )


def test_write_table_body_headers():
    assert (
        "\\textbf{A} & \\textbf{B}\\\\\n"
        "\\hline\n"
        "1 & foo\\\\\n"
        "2 & bar\\\\\n"
        == latex.write_table_body(
            ((1, "foo"), (2, "bar")),
            headers=("A", "B"),
        )
    )


def test_write_table_body_headers_no_hline():
    assert (
        "\\textbf{A} & \\textbf{B}\\\\\n"
        "1 & foo\\\\\n"
        "2 & bar\\\\\n"
        == latex.write_table_body(
            ((1, "foo"), (2, "bar")),
            headers=("A", "B"),
            hline_after_header=False,
        )
    )


def test_write_table_body_headers_no_fmt():
    assert (
        "A & B\\\\\n"
        "\\hline\n"
        "1 & foo\\\\\n"
        "2 & bar\\\\\n"
        == latex.write_table_body(
            ((1, "foo"), (2, "bar")),
            headers=("A", "B"),
            header_fmt=lambda x: x,
        )
    )


def test_write_table_body_hlines():
    assert (
        "\\hline\n"
        "1 & foo\\\\\n"
        "2 & bar\\\\\n"
        == latex.write_table_body(
            ((1, "foo"), (2, "bar")),
            hline_before=True,
        )
    )
    assert (
        "1 & foo\\\\\n"
        "2 & bar\\\\\n"
        "\\hline\n"
        == latex.write_table_body(
            ((1, "foo"), (2, "bar")),
            hline_after=True,
        )
    )


# table()
def test_table():
    assert (
        "\\begin{tabular}{lr}\n"
        "\\toprule\n"
        "   0 &  1 \\\\\n"
        "\\midrule\n"
        " foo &  1 \\\\\n"
        " bar &  2 \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n" == latex.table((("foo", 1), ("bar", 2)))
    )


def test_table_columns():
    assert (
        "\\begin{tabular}{lr}\n"
        "\\toprule\n"
        "type &  value \\\\\n"
        "\\midrule\n"
        " foo &      1 \\\\\n"
        " bar &      2 \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        == latex.table(
            (("foo", 1), ("bar", 2)),
            columns=("type", "value"),
        )
    )


def test_table_bad_columns():
    with test.Raises(latex.Error):
        latex.table(
            (("foo", 1), ("bar", 2)),
            columns=("type", "value", "too", "many", "values"),
        )


def test_table_bad_rows():
    with test.Raises(latex.Error):
        latex.table((("foo", 1), ("bar", 2), ("car",)))


if __name__ == "__main__":
    test.Main()
