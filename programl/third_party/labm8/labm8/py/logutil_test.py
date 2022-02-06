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
"""Unit tests for //labm8/logutil.py."""
import datetime
import pathlib
import tempfile

import pytest
from labm8.py import app, labdate, logutil, test
from labm8.py.internal import logging_pb2

FLAGS = app.FLAGS


def test_ABSL_LOGGING_PREFIX_RE_match():
    """Test that absl logging regex matches a log line."""
    m = logutil.ABSL_LOGGING_LINE_RE.match(
        "I0527 23:14:18.903151 140735784891328 log_to_file.py:31] Hello, info!",
    )
    assert m
    assert m.group("lvl") == "I"
    assert m.group("timestamp") == "0527 23:14:18.903151"
    assert m.group("thread_id") == "140735784891328"
    assert m.group("filename") == "log_to_file.py"
    assert m.group("lineno") == "31"
    assert m.group("contents") == "Hello, info!"


def test_ABSL_LOGGING_PREFIX_RE_not_match():
    """Test that absl logging regex doesn't match a line."""
    m = logutil.ABSL_LOGGING_LINE_RE.match("Hello world!")
    assert not m


# DatetimeFromAbslTimestamp() tests.


def test_DatetimeFromAbslTimestamp():
    dt = logutil.DatetimeFromAbslTimestamp("0527 23:14:18.903151")
    assert dt.year == datetime.datetime.utcnow().year
    assert dt.month == 5
    assert dt.day == 27
    assert dt.hour == 23
    assert dt.minute == 14
    assert dt.second == 18
    assert dt.microsecond == 903151


# ConvertAbslLogToProtos() tests.


def test_ConvertAbslLogToProtos_empty_input():
    assert not logutil.ConertAbslLogToProtos("")


def test_ConvertAbslLogToProtos_num_records():
    p = logutil.ConertAbslLogToProtos(
        """\
I0527 23:14:18.903151 140735784891328 log_to_file.py:31] Hello, info!
W0527 23:14:18.903151 140735784891328 log_to_file.py:31] Hello, warning!
I0527 23:14:18.903151 140735784891328 log_to_file.py:31] Hello ...

multiline!
E0527 23:14:18.903151 1407 log_to_file.py:31] Hello, error!
F0527 23:14:18.903151 1 log_to_file.py:31] Hello, fatal!
"""
    )
    assert 5 == len(p)


def test_ConvertAbslLogToProtos_levels():
    p = logutil.ConertAbslLogToProtos(
        """\
I0527 23:14:18.903151 140735784891328 log_to_file.py:31] Hello, info!
W0527 23:14:18.903151 140735784891328 log_to_file.py:31] Hello, warning!
I0527 23:14:18.903151 140735784891328 log_to_file.py:31] Hello ...

multiline!
E0527 23:14:18.903151 1407 log_to_file.py:31] Hello, error!
F0527 23:14:18.903151 1 log_to_file.py:31] Hello, fatal!
"""
    )
    assert p[0].level == logging_pb2.LogRecord.INFO
    assert p[1].level == logging_pb2.LogRecord.WARNING
    assert p[2].level == logging_pb2.LogRecord.INFO
    assert p[3].level == logging_pb2.LogRecord.ERROR
    assert p[4].level == logging_pb2.LogRecord.FATAL


MS_IN_YEAR = 1000 * 60 * 60 * 24 * 365


def test_ConvertAbslLogToProtos_date_unix_epoch_ms():
    """Test that dates are converted correctly."""
    p = logutil.ConertAbslLogToProtos(
        """\
I0527 23:14:18.903151 140735784891328 log_to_file.py:31] Hello, info!
"""
    )
    dt = labdate.DatetimeFromMillisecondsTimestamp(p[0].date_unix_epoch_ms)
    assert dt.year == datetime.datetime.utcnow().year
    assert dt.month == 5
    assert dt.day == 27
    assert dt.hour == 23
    assert dt.minute == 14
    assert dt.second == 18
    # Microsecond precision has been reduced to millisecond.
    assert dt.microsecond == 903000


def test_ConvertAbslLogToProtos_levels():
    p = logutil.ConertAbslLogToProtos(
        """\
I0527 23:14:18.903151 140735784891328 log_to_file.py:31] Hello, info!
W0527 23:14:18.903151 140735784891328 log_to_file.py:31] Hello, warning!
I0527 23:14:18.903151 140735784891328 log_to_file.py:31] Hello ...

multiline!
E0527 23:14:18.903151 1407 log_to_file.py:31] Hello, error!
F0527 23:14:18.903151 1 log_to_file.py:31] Hello, fatal!
"""
    )
    assert p[0].thread_id == 140735784891328
    assert p[1].thread_id == 140735784891328
    assert p[2].thread_id == 140735784891328
    assert p[3].thread_id == 1407
    assert p[4].thread_id == 1


def test_ConvertAbslLogToProtos_line_number():
    p = logutil.ConertAbslLogToProtos(
        """\
I0527 23:14:18.903151 140735784891328 log_to_file.py:31] Hello, info!
W0527 23:14:18.903151 140735784891328 log_to_file.py:31] Hello, warning!
I0527 23:14:18.903151 140735784891328 log_to_file.py:31] Hello ...

multiline!
E0527 23:14:18.903151 1407 log_to_file.py:31] Hello, error!
F0527 23:14:18.903151 1 log_to_file.py:31] Hello, fatal!
"""
    )
    assert p[0].line_number == 31
    assert p[1].line_number == 31
    assert p[2].line_number == 31
    assert p[3].line_number == 31
    assert p[4].line_number == 31


def test_ConvertAbslLogToProtos_message():
    p = logutil.ConertAbslLogToProtos(
        """\
I0527 23:14:18.903151 140735784891328 log_to_file.py:31] Hello, info!
W0527 23:14:18.903151 140735784891328 log_to_file.py:31] Hello, warning!
I0527 23:14:18.903151 140735784891328 log_to_file.py:31] Hello ...

multiline!
E0527 23:14:18.903151 1407 log_to_file.py:31] Hello, error!
F0527 23:14:18.903151 1 log_to_file.py:31] Hello, fatal!
I0527 23:14:18.903151 140735784891328 log_to_file.py:31] Goodbye ...
multiline!
"""
    )
    assert p[0].message == "Hello, info!"
    assert p[1].message == "Hello, warning!"
    assert p[2].message == "Hello ...\n\nmultiline!"
    assert p[3].message == "Hello, error!"
    assert p[4].message == "Hello, fatal!"
    assert p[5].message == "Goodbye ...\nmultiline!"


# StartTeeLogsToFile() and StopTeeLogsToFile() tests.


def test_TeeLogsToFile_dir_not_found():
    """Test that FileNotFoundError is raised if log_dir does not exist"""
    with tempfile.TemporaryDirectory() as d:
        with test.Raises(FileNotFoundError) as e_info:
            logutil.StartTeeLogsToFile("test", pathlib.Path(d) / "notadir")
        assert "Log directory not found: '{d}/notadir'"


def test_TeeLogsToFile(capsys):
    """Test that StartTeeLogs also logs to file, and StopTeeLogs prevents that."""
    with tempfile.TemporaryDirectory() as d:
        FLAGS.logtostderr = True
        app.Log(1, "This is not going in a file")
        logutil.StartTeeLogsToFile("test", d)
        app.Log(1, "Hello, file!")
        logutil.StopTeeLogsToFile()
        app.Log(1, "This is not going in a file")
        # Test file contents.
        with open(pathlib.Path(d) / "test.INFO") as f:
            lines = f.read().rstrip().split("\n")
            assert len(lines) == 1
            # There is the log formatting bumpf in the line.
            assert "Hello, file!" in lines[0]
        out, err = capsys.readouterr()
        assert not out
        # Test stderr contents.
        lines = err.rstrip().split("\n")
        assert len(lines) == 3
        assert "This is not going in a file" in lines[0]
        assert "Hello, file!" in lines[1]
        assert "This is not going in a file" in lines[2]


def test_TeeLogsToFile_contextmanager(capsys):
    """Test that contextmanager temporarily also logs to file."""
    with tempfile.TemporaryDirectory() as d:
        FLAGS.logtostderr = True
        app.Log(1, "This is not going in a file")
        with logutil.TeeLogsToFile("test", d):
            app.Log(1, "Hello, file!")
        app.Log(1, "This is not going in a file")
        # Test file contents.
        with open(pathlib.Path(d) / "test.INFO") as f:
            lines = f.read().rstrip().split("\n")
            assert len(lines) == 1
            assert "Hello, file!" in lines[0]
        out, err = capsys.readouterr()
        assert not out
        # Test stderr contents.
        lines = err.rstrip().split("\n")
        assert len(lines) == 3
        assert "This is not going in a file" in lines[0]
        assert "Hello, file!" in lines[1]
        assert "This is not going in a file" in lines[2]


if __name__ == "__main__":
    test.Main()
