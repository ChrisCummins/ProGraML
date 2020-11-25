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
"""Unit tests for //labm8/py:decorators.py."""
import time

import pytest

from programl.util.py import decorators
from tests.test_main import main


class DummyClass(object):
    def __init__(self):
        self.memoized_property_run_count = 0

    @decorators.memoized_property
    def memoized_property(self):
        self.memoized_property_run_count += 1
        # In "real world" usage, this would be an expensive computation who's result
        # we would like to memoize.
        return 5


def test_memoized_property_value():
    """Test that memoized property returns expected value."""
    c = DummyClass()
    assert c.memoized_property == 5


def test_memoized_property_run_count():
    """Test that repeated access to property returns memoized value."""
    c = DummyClass()
    _ = c.memoized_property
    _ = c.memoized_property
    _ = c.memoized_property
    assert c.memoized_property_run_count == 1


def test_timeout_timeout_not_raised():
    """Test that decorated function doesn't raise exception."""

    @decorators.timeout(seconds=1)
    def Func() -> int:
        """Function under test."""
        return 5

    assert Func() == 5


def test_timeout_timeout_raised():
    """Test that decorated function doesn't raise exception."""

    @decorators.timeout(seconds=1)
    def Func() -> int:
        """Function under test."""
        time.sleep(10)
        return 5

    with pytest.raises(TimeoutError):
        Func()


if __name__ == "__main__":
    main()
