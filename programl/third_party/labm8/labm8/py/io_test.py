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
"""Unit tests for //labm8/py:text."""
import re
from io import StringIO

import pytest
from labm8.py import app, io, test

FLAGS = app.FLAGS


# colourise()
def test_colourise():
    assert "\033[91mHello, World!\033[0m" == io.colourise(
        io.Colours.RED,
        "Hello, World!",
    )


# printf()
def test_printf():
    out = StringIO()
    io.printf(io.Colours.RED, "Hello, World!", file=out)
    assert "\033[91mHello, World!\033[0m" == out.getvalue().strip()


# pprint()
def test_pprint():
    out = StringIO()
    io.pprint({"foo": 1, "bar": "baz"}, file=out)
    assert '{\n  "bar": "baz",\n  "foo": 1\n}' == out.getvalue().strip()


# info()
def test_info():
    out = StringIO()
    io.info("foo", file=out)
    assert "INFO" == re.search("INFO", out.getvalue()).group(0)


# debug()
def test_debug():
    out = StringIO()
    io.debug("foo", file=out)
    assert "DEBUG" == re.search("DEBUG", out.getvalue()).group(0)


# warn()
def test_warn():
    out = StringIO()
    io.warn("foo", file=out)
    assert "WARN" == re.search("WARN", out.getvalue()).group(0)


# error()
def test_error():
    out = StringIO()
    io.error("foo", file=out)
    assert "ERROR" == re.search("ERROR", out.getvalue()).group(0)


# fatal()
def test_fatal():
    out = StringIO()
    with test.Raises(SystemExit) as ctx:
        io.fatal("foo", file=out)
    assert ctx.value.code == 1
    assert "ERROR" == re.search("ERROR", out.getvalue()).group(0)
    assert "fatal" == re.search("fatal", out.getvalue()).group(0)


def test_fatal_status():
    out = StringIO()
    with test.Raises(SystemExit) as ctx:
        io.fatal("foo", file=out, status=10)
    assert ctx.value.code == 10
    assert "ERROR" == re.search("ERROR", out.getvalue()).group(0)
    assert "fatal" == re.search("fatal", out.getvalue()).group(0)


# prof()
def test_prof():
    out = StringIO()
    io.prof("foo", file=out)
    assert "PROF" == re.search("PROF", out.getvalue()).group(0)


if __name__ == "__main__":
    test.Main()
