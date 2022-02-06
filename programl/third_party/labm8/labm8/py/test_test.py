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
"""Unit tests for //labm8/py:test."""
import os
import pathlib
import random
import sys
import tempfile

from labm8.py import app, test

FLAGS = app.FLAGS

MODULE_UNDER_TEST = None  # No coverage.


# The //:conftest is included implicitly when you depend on //labm8/py:test.
def test_tempdir_fixture_directory_exists(tempdir: pathlib.Path):
    """Test that tempdir fixture returns a directory."""
    assert tempdir.is_dir()


def test_tempdir_fixture_directory_is_empty(tempdir: pathlib.Path):
    """Test that tempdir fixture returns an empty directory."""
    assert not list(tempdir.iterdir())


# Although the 'tempdir' fixture was defined in //:conftest, it can be
# overriden. This overiding fixture will be used for all of the tests in this
# file.
@test.Fixture(scope="function")
def tempdir() -> pathlib.Path:
    """Override the tempdir fixture in //:conftest."""
    with tempfile.TemporaryDirectory(prefix="phd_fixture_override_") as d:
        yield pathlib.Path(d)


def test_tempdir_fixture_overriden(tempdir: pathlib.Path):
    """Test that the overriden test fixture is used, not the one in conftest."""
    assert tempdir.name.startswith("phd_fixture_override_")


@test.SlowTest(reason="This is an example")
def test_mark_slow():
    """A test that is skipped when --test_skip_slow."""
    pass


@test.Flaky(max_runs=100, min_passes=2, reason="Test is nondeterministic.")
def test_mark_flaky():
    """A test which is flaky is one where there is (legitimate) reason for it to
    fail, e.g. because a timeout may or may not trigger depending on system load.
    """
    assert random.random() <= 0.5


@test.Flaky(
    max_runs=100,
    min_passes=2,
    expected_exception=IndexError,
    reason="Test is nondeterministic.",
)
def test_mark_flaky_with_expected_exception():
    """Test that expected_exception can be used to trigger re-runs."""
    if random.random() > 0.5:
        raise IndexError("woops!")


@test.XFail(reason="Test is designed to fail.")
@test.Flaky(expected_exception=IndexError, reason="Test is nondeterministic.")
def test_mark_flaky_with_invalid_expected_exception():
    """Test that only expected_exception triggers a re-run."""
    raise TypeError("woops!")


@test.LinuxTest()
def test_that_only_runs_on_linux():
    """Test that executes only on Linux."""
    pass


@test.MacOsTest()
def test_that_only_runs_on_macos():
    """Test that executes only on MacOs."""
    pass


@test.MacOsTest()
@test.LinuxTest()
def test_that_runs_on_linux_or_macos():
    """Test that will execute both on Linux and MacOS."""
    pass


def test_captured_stdout():
    """A test which prints to stdout."""
    print("This message is captured, unless run with --notest_capture_output")


def test_captured_stderr():
    """A test which prints to stderr."""
    print(
        "This message is captured, unless run with --notest_capture_output",
        file=sys.stderr,
    )


def test_captured_logging_info():
    """A test which prints to app.Log"""
    app.Log(1, "This message is captured unless run with --notest_capture_output")


def test_captured_logging_debug():
    """A test which prints to app.Log"""
    app.Log(2, "This message is captured unless run with --notest_capture_output")


def test_captured_logging_warning():
    """A test which prints to app.Warning"""
    app.Warning(
        "This message is captured unless run with --notest_capture_output",
    )


# Fixture tests.


def test_Fixture_missing_scope():
    with test.Raises(TypeError) as e_ctx:

        @test.Fixture()
        def fixture_Foo() -> int:
            return 1

    assert str(e_ctx.value) == "Test fixture must specify a scope"


@test.Fixture(scope="function")
def foo() -> str:
    """A test fixture which returns a string."""
    return "foo"


def test_Fixture_foo(foo: str):
    """Test that fixture returns expected value."""
    assert foo == "foo"


@test.Fixture(scope="function", params=[1, 2, 3])
def fixture_with_param(request) -> int:
    """A parametrized test fixture."""
    return request.param


def test_Fixture_params(fixture_with_param: int):
    """Test that parameterized test fixtures work."""
    assert fixture_with_param in {1, 2, 3}


# Raises tests.


def test_Raises_expected_error_is_raised():
    with test.Raises(ValueError) as e_ctx:
        raise ValueError("Foo")
    assert str(e_ctx.value) == "Foo"


@test.XFail(reason="Test is designed to fail")
def test_Raises_expected_error_is_not_raises():
    """A test fails if the expected exception is not raised."""
    with test.Raises(AssertionError):
        with test.Raises(ValueError):
            pass


@test.Parametrize("a", (1, 2, 3))
@test.Parametrize("b", (4, 5))
def test_Parameterize(a: int, b: int):
    assert a + b > 4


@test.XFail(reason="Test is designed to fail")
def test_Fail():
    test.Fail("Force a failed test here.")


def test_TemporaryEnv():
    os.environ["ANIMAL"] = "a dog"
    with test.TemporaryEnv() as env:
        assert env["ANIMAL"] == "a dog"
        env["ANIMAL"] = "a cat"
        assert os.environ["ANIMAL"] == "a cat"

    assert os.environ["ANIMAL"] == "a dog"


def test_XML_OUTPUT_FILE_is_not_set():
    """Test that XML_OUTPUT_FILE is unset, either because:
    * we're not in a bazel test environment.
    * it has been unset by test.RunPytestOnFileOrDie().
    """
    assert not os.environ.get("XML_OUTPUT_FILE")


if __name__ == "__main__":
    test.Main()
