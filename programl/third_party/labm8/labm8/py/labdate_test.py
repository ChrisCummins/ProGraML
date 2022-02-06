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
"""Unit tests for //labm8/py:labdate."""
import pytest
from labm8.py import app, labdate, test

FLAGS = app.FLAGS


def test_GetUtcMillisecondsNow_millisecond_precision():
    # Test that milliseconds datetimes have no microseconds.
    now = labdate.GetUtcMillisecondsNow()
    assert not now.microsecond % 1000


def test_MillisecondsTimestamp_invalid_argument():
    with test.Raises(TypeError):
        labdate.MillisecondsTimestamp("not a date")


def test_DatetimeFromMillisecondsTimestamp_default_argument():
    """Current time is used if no timestamp provided."""
    assert labdate.DatetimeFromMillisecondsTimestamp()


def test_DatetimeFromMillisecondsTimestamp_invalid_argument():
    with test.Raises(TypeError):
        labdate.DatetimeFromMillisecondsTimestamp("not a timestamp")


def test_DatetimeFromMillisecondsTimestamp_negative_int():
    with test.Raises(ValueError):
        labdate.DatetimeFromMillisecondsTimestamp(-1)


def test_timestamp_datetime_equivalence():
    date_in = labdate.GetUtcMillisecondsNow()
    timestamp = labdate.MillisecondsTimestamp(date_in)
    date_out = labdate.DatetimeFromMillisecondsTimestamp(timestamp)
    assert date_in == date_out


def test_default_timestamp_datetime_equivalence():
    now = labdate.GetUtcMillisecondsNow()
    timestamp = labdate.MillisecondsTimestamp()
    date_out = labdate.DatetimeFromMillisecondsTimestamp(timestamp)
    assert now.date() == date_out.date()


if __name__ == "__main__":
    test.Main()
