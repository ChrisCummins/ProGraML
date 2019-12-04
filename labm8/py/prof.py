# Copyright 2014-2019 Chris Cummins <chrisc.101@gmail.com>.
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
"""Profiling API for timing critical paths in code.
"""
import contextlib
import csv
import datetime
import inspect
import os
import pathlib
import sys
import time
import typing
from typing import Optional

from labm8.py import app
from labm8.py import humanize
from labm8.py import labdate
from labm8.py import labtypes
from labm8.py import system

_TIMERS = {}


def is_enabled():
  return os.environ.get("PROFILE") is not None


def enable():
  os.environ["PROFILE"] = "1"


def disable():
  os.environ.pop("PROFILE", None)


def isrunning(name):
  """
  Check if a timer is running.

  Arguments:

      name (str, optional): The name of the timer to check.

  Returns:

      bool: True if timer is running, else False.
  """
  return name in _TIMERS


def start(name):
  """
  Start a new profiling timer.

  Arguments:

      name (str, optional): The name of the timer to create. If no
        name is given, the resulting timer is anonymous. There can
        only be one anonymous timer.
      unique (bool, optional): If true, then ensure that timer name
        is unique. This is to prevent accidentally resetting an
        existing timer.

  Returns:

      bool: Whether or not profiling is enabled.
  """
  if is_enabled():
    _TIMERS[name] = time.time()
  return is_enabled()


def stop(name, file=sys.stderr):
  """
  Stop a profiling timer.

  Arguments:

      name (str): The name of the timer to stop. If no name is given, stop
          the global anonymous timer.

  Returns:

      bool: Whether or not profiling is enabled.

  Raises:

      KeyError: If the named timer does not exist.
  """
  if is_enabled():
    elapsed = time.time() - _TIMERS[name]
    if elapsed > 60:
      elapsed_str = "{:.1f} m".format(elapsed / 60)
    elif elapsed > 1:
      elapsed_str = "{:.1f} s".format(elapsed)
    else:
      elapsed_str = "{:.1f} ms".format(elapsed * 1000)

    del _TIMERS[name]
    print("[prof]", name, elapsed_str, file=file)
  return is_enabled()


def profile(fun, *args, **kwargs):
  """
  Profile a function.
  """
  timer_name = kwargs.pop("prof_name", None)

  if not timer_name:
    module = inspect.getmodule(fun)
    c = [module.__name__]
    parentclass = labtypes.get_class_that_defined_method(fun)
    if parentclass:
      c.append(parentclass.__name__)
    c.append(fun.__name__)
    timer_name = ".".join(c)

  start(timer_name)
  ret = fun(*args, **kwargs)
  stop(timer_name)
  return ret


def timers():
  """
  Iterate over all timers.

  Returns:
      Iterable[str]: An iterator over all time names.
  """
  for name in _TIMERS:
    yield name


class ProfileTimer(object):
  """A profiling timer."""

  def __init__(self):
    self.start: datetime.datetime = datetime.datetime.utcnow()
    self.end: Optional[datetime.datetime] = None

  def Stop(self):
    if self.end:
      return
    self.end = datetime.datetime.utcnow()

  @property
  def elapsed(self) -> float:
    if self.end:
      return (self.end - self.start).total_seconds()
    else:
      return (datetime.datetime.utcnow() - self.start).total_seconds()

  @property
  def elapsed_ms(self) -> int:
    return int(round(self.elapsed * 1000))

  def __repr__(self):
    return humanize.Duration(self.elapsed)


@app.skip_log_prefix
@contextlib.contextmanager
def Profile(
  name: typing.Union[str, typing.Callable[[int], str]] = "",
  print_to: typing.Callable[[str], None] = lambda msg: app.Log(1, msg),
) -> ProfileTimer:
  """A context manager which prints the elapsed time upon exit.

  Args:
    name: The name of the task being profiled. A callback may be provided which
      is called at task completion with the elapsed duration in seconds as its
      argument.
    print_to: The function to print the result to.
  """
  name = name or "completed"
  timer = ProfileTimer()
  yield timer
  timer.Stop()
  if callable(name):
    name = name(timer.elapsed)
  print_to(f"{name} in {timer}")


@contextlib.contextmanager
def ProfileToFile(file_object, name: str = ""):
  """A context manager which prints profiling output to a file.

  Args:
    file_object: A file object to write
    name: The name of the task being profiled.
  """

  def _WriteToFile(message: str):
    """Print message to file, appending a newline."""
    file_object.write(f"{message}\n")

  yield Profile(name=name, print_to=_WriteToFile)


class ProfilingEvent(object):
  def __init__(self, start_time: int, name: str):
    self.start_time = start_time
    self.name = name


@contextlib.contextmanager
def ProfileToStdout(name: str = ""):
  """A context manager which prints the elapsed time to stdout on exit.

  Args:
    name: The name of the task being profiled.
  """
  with Profile(name, print_to=print):
    yield


class AutoCsvProfiler(object):
  def __init__(self, directory: pathlib.Path, name: str = "profile"):
    self._directory = pathlib.Path(directory)
    if not self._directory.is_dir():
      raise ValueError(f"Directory not found: {directory}")
    self._name = name

    # Create the name of the logfile now, so that is timestamped to the start of
    # execution.
    timestamp = labdate.MillisecondsTimestamp()
    log_name = ".".join([self._name, system.HOSTNAME, str(timestamp), "csv"])
    self._path = self._directory / log_name

    with self._writer() as writer:
      writer.writerow(
        ("Start Time (ms since UNIX epoch)", "Elapsed Time (ms)", "Event"),
      )

  @contextlib.contextmanager
  def Profile(self, event_name: str = ""):
    """A context manager which prints the elapsed time upon exit.

    Args:
      event_name: The name of the event being profiled.
    """
    event = ProfilingEvent(labdate.MillisecondsTimestamp(), event_name)
    yield event
    elapsed = labdate.MillisecondsTimestamp() - event.start_time
    with self._writer() as writer:
      writer.writerow((event.start_time, elapsed, event.name))

  @contextlib.contextmanager
  def _writer(self):
    with open(self.path, "a") as f:
      writer = csv.writer(f)
      yield writer

  @property
  def path(self) -> pathlib.Path:
    return self._path
