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
"""Unit tests for //labm8/py:system."""
import getpass
import os
import pathlib
import socket
import tempfile

import pytest
from labm8.py import app, fs, system, test

FLAGS = app.FLAGS


@test.Fixture(scope="function")
def tempfile_path() -> str:
    """Test fixture which returns the path to a temporary file."""
    with tempfile.NamedTemporaryFile(prefix="phd_test_") as f:
        yield f.name


def test_hostname():
    hostname = socket.gethostname()
    assert hostname == system.HOSTNAME
    assert hostname == system.HOSTNAME


def test_username():
    username = getpass.getuser()
    assert username == system.USERNAME
    assert username == system.USERNAME


def test_uid():
    uid = os.getuid()
    assert uid == system.UID
    assert uid == system.UID


def test_pid():
    pid = os.getpid()
    assert pid == system.PID
    assert pid == system.PID


# ScpError
def test_ScpError():
    err = system.ScpError("out", "err")
    assert "out" == err.out
    assert "err" == err.err
    assert "out\nerr" == err.__repr__()
    assert "out\nerr" == str(err)


# Subprocess()
def test_subprocess_stdout():
    p = system.Subprocess(["echo Hello"], shell=True)
    ret, out, err = p.run()
    assert not ret
    assert out == "Hello\n"
    assert not err


def test_subprocess_stderr():
    p = system.Subprocess(["echo Hello >&2"], shell=True)
    ret, out, err = p.run()
    assert not ret
    assert err == "Hello\n"
    assert not out


def test_subprocess_timeout():
    p = system.Subprocess(["sleep 10"], shell=True)
    with test.Raises(system.SubprocessError):
        p.run(timeout=0.1)


def test_subprocess_timeout_pass():
    p = system.Subprocess(["true"], shell=True)
    ret, out, err = p.run(timeout=0.1)
    assert not ret


# run()
def test_run():
    assert system.run(["true"]) == (0, None, None)
    assert system.run(["false"]) == (1, None, None)


def test_run_timeout():
    with test.Raises(system.SubprocessError):
        system.run(["sleep 10"], timeout=0.1, shell=True)
    with test.Raises(system.SubprocessError):
        system.run(["sleep 10"], timeout=0.1, num_retries=2, shell=True)


# echo()
def test_echo(tempdir: pathlib.Path):
    file_path = tempdir / "file.txt"
    system.echo("foo", file_path)
    assert fs.read(file_path) == ["foo"]
    system.echo("", file_path)
    assert fs.read(file_path) == [""]


def test_echo_append(tempdir: pathlib.Path):
    file_path = tempdir / "file.txt"
    system.echo("foo", file_path)
    system.echo("bar", file_path, append=True)
    assert fs.read(file_path) == ["foo", "bar"]


def test_echo_kwargs(tempdir: pathlib.Path):
    file_path = tempdir / "file.txt"
    system.echo("foo", file_path, end="_")
    assert fs.read(file_path) == ["foo_"]


# sed()
def test_sed(tempdir: pathlib.Path):
    file_path = tempdir / "file.txt"
    system.echo("Hello, world!", file_path)
    system.sed("Hello", "Goodbye", file_path)
    assert ["Goodbye, world!"] == fs.read(file_path)
    system.sed("o", "_", file_path)
    assert ["G_odbye, world!"] == fs.read(file_path)
    system.sed("o", "_", file_path, "g")
    assert ["G__dbye, w_rld!"] == fs.read(file_path)


def test_sed_fail_no_file(tempdir: pathlib.Path):
    with test.Raises(system.SubprocessError):
        system.sed("Hello", "Goodbye", tempdir / "not_a_file.txt")


# which()
def test_which():
    assert system.which("sh") in {"/bin/sh", "/usr/bin/sh"}
    assert not system.which("not-a-real-command")


def test_which_path():
    assert system.which("sh", path=("/usr", "/bin")) == "/bin/sh"
    assert not system.which("sh", path=("/dev",))
    assert not system.which("sh", path=("/not-a-real-path",))
    assert not system.which("not-a-real-command", path=("/bin",))


def test_isprocess():
    assert system.isprocess(0)
    assert system.isprocess(os.getpid())
    MAX_PROCESSES = 4194303  # OS-dependent. This value is for Linux
    assert not system.isprocess(MAX_PROCESSES + 1)


def test_exit():
    with test.Raises(SystemExit) as ctx:
        system.exit(0)
    assert ctx.value.code == 0
    with test.Raises(SystemExit) as ctx:
        system.exit(1)
    assert ctx.value.code == 1


def test_ProcessFileAndReplace_ok(tempfile_path: str):
    """Test ProcessFileAndReplace with a callback which reverses a file."""
    with open(tempfile_path, "w") as f:
        f.write("Hello, world!")

    def ReverseFile(a: str, b: str):
        with open(a) as af:
            with open(b, "w") as bf:
                bf.write("".join(reversed(af.read())))

    system.ProcessFileAndReplace(tempfile_path, ReverseFile)

    with open(tempfile_path) as f:
        output = f.read()

    assert output == "!dlrow ,olleH"


def test_ProcessFileAndReplace_exception(tempfile_path: str):
    """Test that file is not modified in case of exception."""
    with open(tempfile_path, "w") as f:
        f.write("Hello, world!")

    def BrokenFunction(a: str, b: str):
        del a
        del b
        raise ValueError("Broken function!")

    # Test that error is propagated.
    with test.Raises(ValueError) as e_ctx:
        system.ProcessFileAndReplace(tempfile_path, BrokenFunction)
    assert str(e_ctx.value) == "Broken function!"

    # Test that file is not modified.
    with open(tempfile_path) as f:
        contents = f.read()
    assert contents == "Hello, world!"


if __name__ == "__main__":
    test.Main()
