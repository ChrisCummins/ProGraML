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
"""Unit tests for //labm8/py:jsonutil."""
import pytest
from labm8.py import app, fs, jsonutil, system, test

FLAGS = app.FLAGS


def test_loads():
    a_str = """{
          "a": 1,  // this has comments
          "b": [1, 2, 3]
      } # end comment
      // begin with comment
      """
    a = jsonutil.loads(a_str)
    assert a == {"a": 1, "b": [1, 2, 3]}


def test_loads_malformed():
    a_str = """bad json {asd,,}"""
    with test.Raises(ValueError):
        jsonutil.loads(a_str)


def test_read_file():
    a_str = """{
          "a": 1,  // this has comments
          "b": [1, 2, 3]
      } # end comment
      // begin with comment
      """
    system.echo(a_str, "/tmp/labm8_py.loaf.json")
    a = jsonutil.read_file("/tmp/labm8_py.loaf.json")
    assert a == {"a": 1, "b": [1, 2, 3]}


def test_read_file_bad_path():
    with test.Raises(fs.File404):
        jsonutil.read_file("/not/a/real/path")
    assert not jsonutil.read_file("/not/a/real/path", must_exist=False)


def test_write_file():
    d1 = {"a": "1", "b": "2"}
    jsonutil.write_file("/tmp/labm8_py.write_file.json", d1)
    d2 = jsonutil.read_file("/tmp/labm8_py.write_file.json")
    fs.rm("/tmp/labm8_py.write_file.json")

    jsonutil.write_file("/tmp/labm8_py.write_file2.json", d1)
    d3 = jsonutil.read_file("/tmp/labm8_py.write_file2.json")
    fs.rm("/tmp/labm8_py.write_file2.json")

    assert d1 == d2 == d3


if __name__ == "__main__":
    test.Main()
