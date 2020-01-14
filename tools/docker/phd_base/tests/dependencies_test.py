"""Tests for importing third-party dependencies that have been known to cause
headaches when running inside docker containers.
"""
from labm8.py import test

FLAGS = test.FLAGS


def test_numpy_import():
  """Test numpy module import."""
  import numpy as np

  assert np.array([1, 2, 3]).sum() == 6


def test_mysql_import():
  """The MySQL python API has some tricky native dependencies.

  This test just attempts to import the MySQL api, which will expose any
  dependency errors.

  https://mysqlclient.readthedocs.io/user_guide.html#some-mysql-examples
  """
  from MySQLdb import _mysql

  del _mysql


if __name__ == "__main__":
  test.Main()
