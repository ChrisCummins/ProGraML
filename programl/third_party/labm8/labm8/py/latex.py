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
"""Utilities for generating LaTeX.
"""
import re
from io import StringIO

import pandas
from labm8.py import io


class Error(Exception):
    """
    Module-level error.
    """

    pass


def escape(text):
    return re.sub(r"(_)", r"\\\g<1>", str(text))


def wrap_bold(text):
    return "\\textbf{" + text + "}"


def write_table_body(
    data,
    output=None,
    headers=None,
    header_fmt=wrap_bold,
    hline_after_header=True,
    hline_before=False,
    hline_after=False,
):
    def _write_row(row):
        output.write(" & ".join(escape(column) for column in row) + "\\\\\n")

    # Determine if we're writing to a file or returning a string.
    isfile = output is not None
    output = output or StringIO()

    # Write hline before body.
    if hline_before:
        output.write("\\hline\n")

    # Write headers.
    if headers:
        _write_row(header_fmt(str(column)) for column in headers)
        # Write hline after headers.
        if hline_after_header:
            output.write("\\hline\n")

    # Write table entries.
    for row in data:
        _write_row(row)

    # Write hline after body.
    if hline_after:
        output.write("\\hline\n")

    return None if isfile else output.getvalue()


def table(rows, columns=None, output=None, data_args={}, **kwargs):
    """
    Return a LaTeX formatted string of "list of list" table data.

    See: http://pandas.pydata.org/pandas-docs/dev/generated/pandas.DataFrame.html

    Requires the "booktabs" package to be included in LaTeX preamble:

        \\usepackage{booktabs}

    Examples:

        >>> fmt.print([("foo", 1), ("bar", 2)])
             0  1
        0  foo  1
        1  bar  2

        >>> fmt.print([("foo", 1), ("bar", 2)], columns=("type", "value"))
          type  value
        0  foo      1
        1  bar      2

    Arguments:

        rows (list of list): Data to format, one row per element,
          multiple columns per row.
        columns (list of str, optional): Column names.
        output (str, optional): Path to output file.
        data_args (dict, optional): Any additional kwargs to pass to
          pandas.DataFrame constructor.
        **kwargs: Any additional arguments to pass to
          pandas.DataFrame.to_latex().

    Returns:

        str: Formatted data as LaTeX table.

    Raises:

        Error: If number of columns (if provided) does not equal
          number of columns in rows; or if number of columns is not
          consistent across all rows.
    """
    # Number of columns.
    num_columns = len(rows[0])

    # Check that each row is the same length.
    for i, row in enumerate(rows[1:]):
        if len(row) != num_columns:
            raise Error(
                "Number of columns in row {i_row} ({c_row}) "
                "does not match number of columns in row 0 ({z_row})".format(
                    i_row=i,
                    c_row=len(row),
                    z_row=num_columns,
                ),
            )

    # Check that (if supplied), number of columns matches number of
    # columns in rows.
    if columns is not None and len(columns) != num_columns:
        raise Error(
            "Number of columns in header ({c_header}) does not "
            "match the number of columns in the data ({c_rows})".format(
                c_header=len(columns),
                c_rows=num_columns,
            ),
        )

    # Default arguments.
    if "index" not in kwargs:
        kwargs["index"] = False

    data_args["columns"] = columns

    string = pandas.DataFrame(list(rows), **data_args).to_latex(**kwargs)
    if output is None:
        return string
    else:
        print(string, file=open(output, "w"))
        io.info("Wrote", output)
