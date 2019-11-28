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
"""Python unit test main entry point.

This project uses pytest runner, with a handful of custom configuration options.
Use the Main() function as the entry point to your test files to run pytest
with the proper arguments.
"""
import contextlib
import inspect
import pathlib
import re
import sys
import tempfile
import typing
from importlib import util as importutil

import pytest

from labm8.py import app

FLAGS = app.FLAGS

app.DEFINE_boolean("test_color", True, "Colorize pytest output.")
app.DEFINE_boolean(
  "test_skip_slow", False, "Skip tests that have been marked slow.",
)
app.DEFINE_boolean(
  "test_skip_benchmarks", False, "Skip tests that are benchmarks."
)
app.DEFINE_integer(
  "test_maxfail",
  1,
  "The maximum number of tests that can fail before execution terminates. "
  "If --test_maxfail=0, all tests will execute.",
)
app.DEFINE_boolean(
  "test_capture_output",
  True,
  "Capture stdout and stderr during test execution.",
)
app.DEFINE_boolean(
  "test_print_durations",
  True,
  "Print the duration of the slowest tests at the end of execution. Use "
  "--test_durations to set the number of tests to print the durations of.",
)
app.DEFINE_integer(
  "test_durations",
  3,
  "The number of slowest tests to print the durations of after execution. "
  "If --test_durations=0, the duration of all tests is printed.",
)
app.DEFINE_string(
  "test_coverage_data_dir",
  None,
  "Run tests with statement coverage and write coverage.py data files to "
  "this directory. The directory is created. Existing files are untouched.",
)


def AbsolutePathToModule(file_path: str) -> str:
  """Determine module name from an absolute path."""
  match = re.match(r".+\.runfiles/phd/(.+)", file_path)
  if match:
    # Strip everything up to the root of the project from the path.
    module = match.group(1)
    # Strip the .py suffix.
    module = module[: -len(".py")]
    # Replace path sep with module sep.
    module = module.replace("/", ".")
    return module
  else:
    raise OSError(f"Could not determine runfiles directory: {file_path}")


def GuessModuleUnderTest(file_path: str) -> typing.Optional[str]:
  """Determine the module under test. Returns None if no module under test."""
  # Load the test module and check for a MODULE_UNDER_TEST attribute. If
  # present, this is the name of the module under test. Valid values for
  # MODULE_UNDER_TEST are a string, e.g. 'labm8.py.app', or None.
  spec = importutil.spec_from_file_location("module", file_path)
  test_module = importutil.module_from_spec(spec)
  spec.loader.exec_module(test_module)
  if hasattr(test_module, "MODULE_UNDER_TEST"):
    return test_module.MODULE_UNDER_TEST

  # If the module under test was not specified, Guess the module name by
  # stripping the '_test' suffix from the name of the test module.
  return AbsolutePathToModule(file_path)[: -len("_test")]


@contextlib.contextmanager
def CoverageContext(
  file_path: str, pytest_args: typing.List[str],
) -> typing.List[str]:

  # Record coverage of module under test.
  module = GuessModuleUnderTest(file_path)
  if not module:
    app.Log(1, "Coverage disabled - no module under test")
    yield pytest_args
    return

  with tempfile.TemporaryDirectory(prefix="phd_test_") as d:
    # If we
    if FLAGS.test_coverage_data_dir:
      datadir = pathlib.Path(FLAGS.test_coverage_data_dir)
      datadir.mkdir(parents=True, exist_ok=True)
    else:
      datadir = pathlib.Path(d)
    # Create a coverage.py config file.
    # See: https://coverage.readthedocs.io/en/coverage-4.3.4/config.html
    config_path = f"{d}/converagerc"
    with open(config_path, "w") as f:
      f.write(
        f"""\
[run]
data_file = {datadir}/.coverage
parallel = True
# disable_warnings =
#   module-not-imported
#   no-data-collected
#   module-not-measured

[report]
ignore_errors = True
# Regexes for lines to exclude from consideration
exclude_lines =
  # Have to re-enable the standard pragma
  pragma: no cover

  # Don't complain about missing debug-only code:
  def __repr__
  if self\.debug

  # Don't complain if tests don't hit defensive assertion code:
  raise AssertionError
  raise NotImplementedError

  # Don't complain if non-runnable code isn't run:
  if 0:
  if __name__ == .__main__.:
"""
      )

    pytest_args += [
      f"--cov={module}",
      f"--cov-config={config_path}",
    ]
    yield pytest_args


def RunPytestOnFileAndExit(
  file_path: str, argv: typing.List[str], capture_output: bool = None
):
  """Run pytest on a file and exit.

  This is invoked by absl.app.RunWithArgs(), and has access to absl flags.

  This function does not return.

  Args:
    file_path: The path of the file to test.
    argv: Positional arguments not parsed by absl. No additional arguments are
      supported.
    capture_output: Whether to capture stdout/stderr when running tests. If
      provided, this value overrides --test_capture_output.
  """
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(" ".join(argv[1:])))

  # Always run with the most verbose logging option.
  app.FLAGS.vmodule += [
    "*=5",
  ]

  # Allow capture_output to override the flags value, if provided.
  if capture_output is not None:
    app.FLAGS.test_capture_output = capture_output

  # Test files must end with _test.py suffix. This is a code style choice, not
  # a hard requirement.
  if not file_path.endswith("_test.py"):
    app.Fatal("File `%s` does not end in suffix _test.py", file_path)

  # Assemble the arguments to run pytest with. Note that the //:conftest file
  # performs some additional configuration not captured here.
  pytest_args = [
    file_path,
    # Run pytest verbosely.
    "-vv",
    "-p",
    "no:cacheprovider",
  ]

  if FLAGS.test_color:
    pytest_args.append("--color=yes")

  if FLAGS.test_maxfail != 0:
    pytest_args.append(f"--maxfail={FLAGS.test_maxfail}")

  if FLAGS.test_skip_benchmarks:
    pytest_args.append("--benchmark-skip")

  # Print the slowest test durations at the end of execution.
  if FLAGS.test_print_durations:
    pytest_args.append(f"--durations={FLAGS.test_durations}")

  # Capture stdout and stderr by default.
  if not FLAGS.test_capture_output:
    pytest_args.append("-s")

  with CoverageContext(file_path, pytest_args) as pytest_args:
    app.Log(1, "Running pytest with arguments: %s", pytest_args)
    ret = pytest.main(pytest_args)
  sys.exit(ret)


def Fixture(scope: str = "", params: typing.Optional[typing.Any] = None):
  """Construct a test fixture.

  This is a wrapper around pytest's fixture which enforces various project-local
  options such as requiring a `scope` argument.

  See: https://docs.pytest.org/en/latest/fixture.html

  Args:
    scope: The scope of the fixture. One of {function, class, package, module}.
    params: An optional list of parameters which will cause multiple invocations
      of the fixture function and all of the tests using it.

  Usage:

      @test.Fixture
      def foo() -> int:
        return 5

      def test_foo(foo: int):
        assert foo == 5
  """
  if not scope:
    raise TypeError(f"Test fixture must specify a scope")

  return pytest.fixture(scope=scope, params=params)


def Raises(expected_exception: typing.Callable):
  """A context manager to wrap code that raises an exception.

  Usage:

      def test_foo():
        with test.Raises(ValueError):
          FunctionThatRaisesError()
  """
  return pytest.raises(expected_exception)


def Flaky(max_runs: int = 5, min_passes: int = 1, expected_exception=None):
  """Mark a test as flaky."""

  def ReRunFilter(err, *args):
    """A function that determines whether to re-run a failed flaky test."""
    del args
    if expected_exception:
      error_class, exception, taceback = err
      return issubclass(error_class, expected_exception)
    else:
      return True

  return pytest.mark.flaky(
    max_runs=max_runs, min_passes=min_passes, rerun_filter=ReRunFilter
  )


def SlowTest(reason: str = ""):
  """Mark a test as slow. Slow tests may be skipped using --test_skip_slow."""
  if not reason:
    raise TypeError("Must provide a reason for slow test.")
  return pytest.mark.slow


def MacOsTest():
  """Mark a test that runs only on macOS. This test will be skipped on linux."""
  return pytest.mark.darwin


def LinuxTest():
  """Mark a test that runs only on linux. This test will be skipped on macOS."""
  return pytest.mark.linux


def XFail(reason: str = ""):
  """Mark a test as expected to fail. An XFail test that *passes* is treated
  as a failure."""
  if not reason:
    raise TypeError("Must provide a reason for XFail test.")
  return pytest.mark.xfail(strict=True, reason=reason)


def Parametrize(arg_name: str, arg_values: typing.Tuple[typing.Any]):
  """Create a parametrized function."""
  return pytest.mark.parametrize(arg_name, arg_values)


def Skip(reason: str = ""):
  """Mark a test as one to skip."""
  if not reason:
    raise TypeError("Must provide a reason to skip test.")
  return pytest.mark.skip(reason=reason)


def SkipIf(condition: bool, reason: str = ""):
  """Skip the test if the condition is met."""
  if not reason:
    raise TypeError("Must provide a reason to conditionally skip a test.")
  return pytest.mark.skipif(condition, reason)


def Fail(reason: str):
  """Mark a test as failed."""
  return pytest.fail(reason)


def Main(capture_output: typing.Optional[bool] = None):
  """Main entry point.

  Args:
    capture_output: Whether to capture stdout/stderr when running tests. If
      provided, this value overrides --test_capture_output.
  """
  # Get the file path of the calling function. This is used to identify the
  # script to run the tests of.
  frame = inspect.stack()[1]
  module = inspect.getmodule(frame[0])
  file_path = module.__file__

  app.RunWithArgs(
    lambda argv: RunPytestOnFileAndExit(
      file_path, argv, capture_output=capture_output
    )
  )
