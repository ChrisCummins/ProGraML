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
"""Unit tests for //labm8/py:prof."""
import os
import re
from io import StringIO

import pytest
from labm8.py import app, prof, test

FLAGS = app.FLAGS


@test.Fixture(scope="function")
def profiling_env() -> None:
    """Create a session for an in-memory SQLite datastore.

    The database is will be empty.

    Returns:
      A Session instance.
    """
    try:
        os.environ["PROFILE"] = "1"
        yield None
    finally:
        os.environ["PROFILE"] = ""


def test_enable_disable():
    assert not prof.is_enabled()
    prof.disable()
    assert not prof.is_enabled()
    prof.enable()
    assert prof.is_enabled()
    prof.disable()
    assert not prof.is_enabled()


def test_named_timer(profiling_env):
    buf = StringIO()

    prof.start("foo")
    prof.stop("foo", file=buf)

    out = buf.getvalue()
    assert " foo " == re.search(" foo ", out).group(0)


def test_named_timer(profiling_env):
    buf = StringIO()

    prof.start("foo")
    prof.start("bar")
    prof.stop("bar", file=buf)

    out = buf.getvalue()
    assert not re.search(" foo ", out)
    assert " bar " == re.search(" bar ", out).group(0)

    prof.stop("foo", file=buf)

    out = buf.getvalue()
    assert " foo " == re.search(" foo ", out).group(0)
    assert " bar " == re.search(" bar ", out).group(0)


def test_stop_twice_error(profiling_env):
    prof.start("foo")
    prof.stop("foo")
    with test.Raises(KeyError):
        prof.stop("foo")


def test_stop_bad_name_error(profiling_env):
    with test.Raises(KeyError):
        prof.stop("not a timer")


def test_profile(profiling_env):
    def test_fn(x, y):
        return x + y

    assert prof.profile(test_fn, 1, 2) == 3


def test_timers(profiling_env):
    x = len(list(prof.timers()))
    prof.start("new timer")
    assert len(list(prof.timers())) == x + 1
    prof.stop("new timer")
    assert len(list(prof.timers())) == x


if __name__ == "__main__":
    test.Main()
