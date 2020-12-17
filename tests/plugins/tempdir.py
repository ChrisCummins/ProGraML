# Copyright 2019-2020 the ProGraML authors.
#
# Contact Chris Cummins <chrisc.101@gmail.com>.
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
import pathlib
import tempfile

import pytest


@pytest.fixture(scope="function")
def tempdir() -> pathlib.Path:
    """A test fixture which yields a temporary directory."""
    with tempfile.TemporaryDirectory(prefix="phd_test_") as d:
        yield pathlib.Path(d)


@pytest.fixture(scope="function")
def tempdir2() -> pathlib.Path:
    """For when a single temporary directory just isn't enough!"""
    with tempfile.TemporaryDirectory(prefix="phd_test_") as d:
        yield pathlib.Path(d)


@pytest.fixture(scope="function")
def tempdir3() -> pathlib.Path:
    """For when a two temporary directories just aren't enough!"""
    with tempfile.TemporaryDirectory(prefix="phd_test_") as d:
        yield pathlib.Path(d)


@pytest.fixture(scope="module")
def module_tempdir() -> pathlib.Path:
    """A test fixture which yields a temporary directory.

    This is the same as tempdir(), except that the directory yielded is the same
    for all tests in a module. Use this when composing a module-level fixture
    which requires a tempdir. For all other uses, the regular tempdir() should
    be suitable.
    """
    with tempfile.TemporaryDirectory(prefix="phd_test_") as d:
        yield pathlib.Path(d)
