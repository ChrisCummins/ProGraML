# Copyright 2014-2019 Chris Cummins <chrisc.101@gmail.com>.
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
"""Utility code for working with pandas."""
import typing

import pandas as pd
import tabulate
from absl import flags as absl_flags

from labm8.py import sqlutil

FLAGS = absl_flags.FLAGS


def QueryToDataFrame(
  session: sqlutil.Session, query: sqlutil.Query
) -> pd.DataFrame:
  """Read query results to a Pandas DataFrame.

  Args:
    session: A database session.
    query: The query to run.

  Returns:
    A Pandas DataFrame.
  """
  return pd.read_sql(query.statement, session.bind)


def ModelToDataFrame(
  session: sqlutil.Session,
  model,
  columns: typing.Optional[typing.List[str]] = None,
  query_identity=lambda q: q,
):
  """Construct and execute a query reads an object's fields to a dataframe.

  Args:
    session: A database session.
    model: A database mapped object.
    columns: A list of column names, where each element is a column mapped to
      the model. If not provided, all column names are used.
    query_identity: A function which takes the produced query and returns a
      query. Use this to implement filtering of the query results.

  Returns:
    A Pandas DataFrame with one column for each field.
  """
  columns = columns or ColumnNames(model)
  query = session.query(*[getattr(model, column) for column in columns])
  df = QueryToDataFrame(session, query_identity(query))
  df.columns = columns
  return df


def RewriteColumn(
  df: pd.DataFrame,
  column: str,
  rewrite: typing.Callable[[typing.Any], typing.Any],
):
  """Rewrite the values in a column in-place."""
  rewrites = []
  for x in df[column]:
    try:
      rewrites.append(rewrite(x))
    except Exception as e:
      raise ValueError(f"Error while rewriting '{x}': {e}")

  df[column] = rewrites


def FormatDataFrameAsAsciiTable(df: pd.DataFrame, **tabulate_args):
  default_args = {
    "headers": "keys",
    "tablefmt": "psql",
  }
  default_args.update(tabulate_args)
  return tabulate.tabulate(df, **default_args)
