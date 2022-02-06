"""Unit tests for //labm8/bazelutil.py."""
import pathlib
import tempfile

import pytest
from labm8.py import app, bazelutil, fs, test

FLAGS = app.FLAGS

# DataPath() tests.


def test_DataPath_path_not_found():
    """Test that FileNotFoundError is raised if the file is not found."""
    with test.Raises(FileNotFoundError) as e_info:
        bazelutil.DataPath("")
    assert f"No such file or directory: ''" in str(e_info)

    with test.Raises(FileNotFoundError) as e_info:
        bazelutil.DataPath("/not/a/real/path")
    assert f"No such file or directory: '/not/a/real/path'" in str(e_info)


def test_DataPath_missing_data_dep():
    """FileNotFoundError is raised if the file exists is not in target data."""
    # The file //labm8/py/test_data/diabetes.csv exists, but is not a data
    # dependency of this test target, so is not found.
    with test.Raises(FileNotFoundError) as e_info:
        bazelutil.DataPath("phd/labm8/py/test_data/diabetes.csv")
    assert (
        "No such file or directory: " "'phd/labm8/py/test_data/diabetes.csv'"
    ) in str(e_info)


def test_DataPath_missing_data_dep_not_must_exist():
    """Path is returned if the file doesn't exist."""
    # The file //labm8/py/test_data/diabetes.csv exists, but is not a data
    # dependency of this test target, so is not found.
    assert bazelutil.DataPath(
        "phd/labm8/py/test_data/diabetes.csv",
        must_exist=False,
    )


def test_DataPath_read_file():
    """Test that DataPath is correct for a known data file."""
    with open(bazelutil.DataPath("phd/labm8/py/test_data/hello_world")) as f:
        assert f.read() == "Hello, world!\n"


def test_DataString_missing_data_dep():
    """FileNotFoundError is raised if the file exists is not in target data."""
    # The file //labm8/py/test_data/diabetes.csv exists, but is not a data
    # dependency of this test target, so is not found.
    with test.Raises(FileNotFoundError) as e_info:
        bazelutil.DataString("phd/labm8/py/test_data/diabetes.csv")


def test_DataString_contents():
    """Test that DataString is correct for a known data file."""
    assert (
        bazelutil.DataString("phd/labm8/py/test_data/hello_world").decode("utf-8")
        == "Hello, world!\n"
    )


def test_DataPath_directory():
    """Test that DataPath returns the path to a directory."""
    assert str(bazelutil.DataPath("phd/labm8/py/test_data")).endswith(
        "phd/labm8/py/test_data",
    )


def test_DataPath_different_working_dir():
    """Test that DataPath is not affected by current working dir."""
    p = bazelutil.DataPath("phd/labm8/py/test_data/hello_world")
    with fs.chdir("/tmp"):
        assert bazelutil.DataPath("phd/labm8/py/test_data/hello_world") == p
    with tempfile.TemporaryDirectory() as d:
        with fs.chdir(d):
            assert bazelutil.DataPath("phd/labm8/py/test_data/hello_world") == p


def test_DataArchive_path_not_found(tempdir: pathlib.Path):
    """Test that FileNotFound raised if path doesn't exist."""
    with test.Raises(FileNotFoundError) as e_ctx:
        bazelutil.DataArchive(tempdir / "a.zip")
    # The exception is raised by bazelutil.DataPath(), so has a slightly different
    # message than the one raised by Archive.__init__().
    assert str(e_ctx.value).startswith("No such file or directory: '")


if __name__ == "__main__":
    test.Main()
