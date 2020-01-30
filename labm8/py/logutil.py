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
"""Utility code for converting between absl logging output and log protos."""
import contextlib
import datetime
import pathlib
import re
import sys
import typing

from absl import logging

from labm8.py import app
from labm8.py import labdate
from labm8.py.internal import logging_pb2

FLAGS = app.FLAGS

# A regular expression to match the components of an absl logging prefix. See:
# https://github.com/abseil/abseil-py/blob/e69e200f680a20c50e0e2cd9e74e9850ff69b856/absl/logging/__init__.py#L554-L583
ABSL_LOGGING_LINE_RE = re.compile(
  r"(?P<lvl>[IWEF])(?P<timestamp>\d\d\d\d \d\d:\d\d:\d\d.\d\d\d\d\d\d) "
  r"(?P<thread_id>\d+) (?P<filename>[^:]+):(?P<lineno>\d+)] "
  r"(?P<contents>.*)",
)

# Convert a single letter absl logging prefix to a LogRecord.LogLevel. Since
# absl logging uses the same prefix for logging.DEBUG and logging.INFO, this
# conversion is lossy, as LogRecord.DEBUG is never returned.
ABSL_LEVEL_TO_LOG_RECORD_LEVEL = {
  "I": logging_pb2.LogRecord.INFO,
  "W": logging_pb2.LogRecord.WARNING,
  "E": logging_pb2.LogRecord.ERROR,
  "F": logging_pb2.LogRecord.FATAL,
}


def DatetimeFromAbslTimestamp(
  timestamp: str, year: int = datetime.datetime.utcnow().year,
) -> datetime.datetime:
  """Convert absl logging timestamp to datetime.

  WARNING: Absl logs do not include the year, so if parsing logs from previous
  years, be sure to set the year argument! The default value assumes the logs
  are from the current year.

  Args:
    timestamp: The string timestamp.
    year: The year, as an integer. E.g. 2019.

  Returns:
    A datetime.
  """
  dt = datetime.datetime.strptime(str(year) + timestamp, "%Y%m%d %H:%M:%S.%f")
  return dt


def ConertAbslLogToProtos(
  logs: str, year: int = datetime.datetime.utcnow().year,
) -> typing.List[logging_pb2.LogRecord]:
  """Convert the output of logging with absl logging to LogRecord protos.

  WARNING: Absl logs do not include the year, so if parsing logs from previous
  years, be sure to set the year argument! The default value assumes the logs
  are from the current year.

  Args:
    logs: The output from logging with absl.
    year: The year, as an integer. E.g. 2019.

  Returns:
    A list of LogRecord messages.
  """
  records = []
  starting_match = None
  lines_buffer = []

  def ConvertOne() -> logging_pb2.LogRecord:
    """Convert the current starting_match and lines_buffer into a LogRecord."""
    if starting_match:
      records.append(
        logging_pb2.LogRecord(
          level=ABSL_LEVEL_TO_LOG_RECORD_LEVEL[starting_match.group("lvl")],
          date_unix_epoch_ms=labdate.MillisecondsTimestamp(
            DatetimeFromAbslTimestamp(
              starting_match.group("timestamp"), year=year,
            ),
          ),
          thread_id=int(starting_match.group("thread_id")),
          file_name=starting_match.group("filename"),
          line_number=int(starting_match.group("lineno")),
          message="\n".join(
            [starting_match.group("contents")] + lines_buffer,
          ).rstrip(),
        ),
      )

  for line in logs.split("\n"):
    m = ABSL_LOGGING_LINE_RE.match(line)
    if m:
      ConvertOne()
      starting_match = None
      lines_buffer = []
      starting_match = m
    elif line and not starting_match:
      raise ValueError(f"Failed to parse logging output at line: '{line}'")
    else:
      lines_buffer.append(line)
  ConvertOne()
  return records


def StartTeeLogsToFile(
  program_name: str = None,
  log_dir: str = None,
  file_log_level: int = logging.DEBUG,
) -> None:
  """Log messages to file as well as stderr.

  Args:
    program_name: The name of the program.
    log_dir: The directory to log to.
    file_log_level: The minimum verbosity level to log to file to.

  Raises:
    FileNotFoundError: If the requested log_dir does not exist.
  """
  if not pathlib.Path(log_dir).is_dir():
    raise FileNotFoundError(f"Log directory not found: '{log_dir}'")
  old_verbosity = logging.get_verbosity()
  logging.set_verbosity(file_log_level)
  logging.set_stderrthreshold(old_verbosity)
  logging.get_absl_handler().start_logging_to_file(program_name, log_dir)
  # The Absl logging handler function start_logging_to_file() sets logtostderr
  # to False. Re-enable whatever value it was before the call.
  FLAGS.logtostderr = False


def StopTeeLogsToFile():
  """Stop logging messages to file as well as stderr."""
  logging.get_absl_handler().flush()
  logging.get_absl_handler().stream = sys.stderr
  FLAGS.logtostderr = True


@contextlib.contextmanager
def TeeLogsToFile(
  program_name: str = None,
  log_dir: str = None,
  file_log_level: int = logging.DEBUG,
):
  """Temporarily enable logging to file.

  Args:
    program_name: The name of the program.
    log_dir: The directory to log to.
    file_log_level: The minimum verbosity level to log to file to.
  """
  try:
    StartTeeLogsToFile(program_name, log_dir, file_log_level)
    yield
  finally:
    StopTeeLogsToFile()
