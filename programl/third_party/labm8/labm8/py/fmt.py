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
"""String formatting utils.
"""
import typing

import pandas as pd


class Error(Exception):
    """
    Module-level error.
    """

    pass


def IndentList(num_spaces: int, strings: typing.List[str]) -> typing.List[str]:
    """Indent each string in a list of strings by a number of spaces.

    Args:
        num_spaces: The number of spacces to indent by. Must be greater than or
          equal to zero.
        strings: The list of strings to indent.

    Returns:
      A list of indented strings.
    """
    prefix = "".join([" "] * num_spaces)
    return [f"{prefix}{string}" for string in strings]


def Indent(num_spaces: int, text: str) -> str:
    """Indent a string by a number of spaces.

    Prepends 'num_spaces' whitespace characters to every line in the given string.

    Args:
      num_spaces: The number of spaces to indent by. Must be greater than or equal
        to zero.
      text: The string to indent.

    Returns:
      The indented string.
    """
    return "\n".join(IndentList(num_spaces, text.split("\n")))


def table(rows, columns=None, output=None, data_args={}, **kwargs) -> str:
    """
    Return a formatted string of "list of list" table data.

    See: http://pandas.pydata.org/pandas-docs/dev/generated/pandas.DataFrame.html

    Examples:
      >>> table([("foo", 1), ("bar", 2)])
           0  1
      0  foo  1
      1  bar  2

      >>> table([("foo", 1), ("bar", 2)], columns=("type", "value"))
        type  value
      0  foo      1
      1  bar      2

    Arguments:
      rows (list of list): Data to format, one row per element,
        multiple columns per row.
      columns (list of str, optional): Column names.
      output (str, optional): Path to output file.
      data_args (dict, optional): Any additional kwargs to pass to
        pd.DataFrame constructor.
      **kwargs: Any additional arguments to pass to
        pd.DataFrame.to_string().

    Returns:
      str: Formatted data as table.

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

    if columns is None:
        # Default parameters.
        if "header" not in kwargs:
            kwargs["header"] = False
    elif len(columns) != num_columns:
        # Check that number of columns matches number of columns in
        # rows.
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

    string = pd.DataFrame(list(rows), **data_args).to_string(**kwargs)
    if output is None:
        return string
    else:
        print(string, file=open(output, "w"))
        print("Wrote", output)
