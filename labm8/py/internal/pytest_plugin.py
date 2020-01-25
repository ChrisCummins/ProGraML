"""This module defines a pytest plugin for labm8."""
import os
import pathlib
import socket
import sys
import tempfile
import typing

import pytest

import build_info
from labm8.py import app

# *WARNING* Flags used in this file are not defined here! They are declared in
# //labm8/py:test.
FLAGS = app.FLAGS

# Test fixtures.

# Note that @pytest.fixture is used here in place of @test.Fixture to break
# a circular dependency between this file and //labm8/py:test.


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


# Pytest configuration.

# The names of platforms which can be used to mark tests.
PLATFORM_NAMES = set("darwin linux win32".split())

# The host names which can be used to mark tests.
HOST_NAMES = set("diana florence".split())


def pytest_configure(config):
  """Programatic configuration of pytest."""
  # Register the custom markers used by labm8.
  config.addinivalue_line("markers", "slow")
  config.addinivalue_line("markers", "darwin")
  config.addinivalue_line("markers", "linux")
  config.addinivalue_line("markers", "win32")
  config.addinivalue_line("markers", "diana")
  config.addinivalue_line("markers", "florence")


def pytest_collection_modifyitems(config, items):
  """A pytest hook to modify the configuration and items to run."""
  del config

  # Fail early and verbosely if the flags cannot be accessed. This is a sign
  # that this file is being used incorrectly. To use this file, you must
  # use labm8.py.test.Main() as the entry point to your tests.
  try:
    FLAGS.test_color
  except AttributeError:
    app.Fatal(
      "Failed to access flags defined in //labm8/py:test. Are you "
      "sure you are running this test using labm8.py.test.Main()?"
    )

  this_platform = sys.platform
  this_host = socket.gethostname()
  slow_skip_marker = pytest.mark.skip(reason="Use --notest_skip_slow to run")

  # Optionally rewrite the file path determined by the pytest collector.
  # The conditions for rewriting the file paths are:
  #   * We are running in a bazel environment.
  #   * The workspace directory determined by build_info exists.
  #
  # The reason for rewriting the file path is that bazel's containerized test
  # execution means that the location of the pytest root directory is not the
  # same as the root directory of the sources being tested. Further, I don't
  # want to set --root_dir to the real workspace root as that seems too invasive
  # of a change to bazel's testing strategy.
  #
  # NOTE: Since this plugin is intended to be used only by labm8.py.test.Main(),
  # it is assumed that all test items come from the same location, so we need
  # only determine the "real" location of a single item.
  #
  # WARNING: It is unclear to me whether rewriting the location will have some
  # unexpected sideffects for bits of pytest that depend on the location. In
  # my experience so far, this rewriting has not caused any problems.
  rewriten_path = None
  workspace = build_info.GetBuildInfo().unsafe_workspace
  if items and os.environ.get("TEST_TARGET") and os.path.isdir(workspace):
    rewriten_path = os.path.relpath(items[0].location[0], workspace)

  for item in items:
    # TODO(cec): Skip benchmarks by default.

    if rewriten_path:
      item.location = (rewriten_path, *item.location[1:])

    # Skip tests if they been marked for an incompatible platform. To mark a
    # test for a platform, wrap the test function with a decorator. Example:
    #
    #   @test.MacOsTest()
    #   def test_will_only_run_on_darwin():
    #     pass
    #
    # Platform decorators can be combined to support multiple platforms.
    supported_platforms = PLATFORM_NAMES.intersection(item.keywords)
    if supported_platforms and this_platform not in supported_platforms:
      skip_msg = f"Skipping `{item.name}` for platforms: {supported_platforms}"
      app.Log(1, skip_msg)
      item.add_marker(pytest.mark.skip(reason=skip_msg))
      continue

    # Skip tests if they have been marked for a specific hostname.
    supported_hosts = HOST_NAMES.intersection(item.keywords)
    if supported_hosts and this_host not in supported_hosts:
      skip_msg = f"Skipping `{item.name}` for hosts: {supported_hosts}"
      app.Log(1, skip_msg)
      item.add_marker(pytest.mark.skip(reason=skip_msg))
      continue

    # Skip tests that have been marked slow unless --notest_skip_slow. To mark
    # a test as slow, wrap the test function with a decorator. Example:
    #
    #   @test.SlowTest(reason='This takes a while')
    #   def test_long_running():
    #     ExpensiveTest()
    #
    # We could achieve the same effect by simple running with pytest with the
    # arguments `-m 'not slow'`, but skipping tests in this manner is silent.
    # Explicitly marking them as skipped, as done here, ensures that the test
    # name still appears in the test output, with a 'skipped' message. This is
    # useful for keeping track of how many tests in a file are *not* being run.
    if FLAGS.test_skip_slow and "slow" in item.keywords:
      app.Log(1, "Skipping `%s` because it is slow", item.name)
      item.add_marker(slow_skip_marker)
      continue


def pytest_report_header(config) -> typing.Union[str, typing.List[str]]:
  """A pytest hook which returns an additional string (or list of strings) to
  print in the header of the test report. See:
  https://docs.pytest.org/en/latest/example/simple.html#adding-info-to-test-report-header
  """
  del config
  return f"phd: {build_info.FormatShortBuildDescription()}"
