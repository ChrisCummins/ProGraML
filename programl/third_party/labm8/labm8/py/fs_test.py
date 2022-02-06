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
"""Unit tests for //labm8/py:fs."""
import os
import pathlib
import stat
import tempfile

from labm8.py import app, fs, system, test

FLAGS = app.FLAGS


# path()
def test_path():
    assert "foo/bar" == fs.path("foo", "bar")
    assert "foo/bar/car" == fs.path("foo/bar", "car")


def test_path_homedir():
    assert os.path.expanduser("~") == fs.path("~")
    assert os.path.join(os.path.expanduser("~"), "foo") == fs.path("~", "foo")


def test_must_exist():
    with tempfile.NamedTemporaryFile(prefix="labm8_") as f:
        assert fs.must_exist(f.name) == f.name
        assert fs.must_exist(fs.dirname(f.name), fs.basename(f.name)) == f.name
    with test.Raises(fs.File404):
        fs.must_exist("/not/a/real/path")


# abspath()
def test_abspath():
    assert os.path.abspath(".") + "/foo/bar" == fs.abspath("foo", "bar")
    assert os.path.abspath(".") + "/foo/bar/car" == fs.abspath("foo/bar", "car")


def test_abspath_homedir():
    assert os.path.expanduser("~") == fs.abspath("~")
    assert os.path.join(os.path.expanduser("~"), "foo") == fs.abspath(
        "~",
        "foo",
    )


# is_subdir()
def test_is_subdir():
    assert fs.is_subdir("/home", "/")
    assert fs.is_subdir("/proc/1", "/proc")
    assert fs.is_subdir("/proc/1", "/proc/1/")
    assert not fs.is_subdir("/proc/3", "/proc/1/")
    assert not fs.is_subdir("/", "/home")


def test_is_subdir_not_subdir():
    assert not fs.is_subdir("/", "/home")


# basename()
def test_basename():
    assert "foo" == fs.basename("foo")
    assert "foo" == fs.basename(fs.abspath("foo"))


def test_dirname():
    assert "" == fs.dirname("foo")
    assert "/tmp" == fs.dirname("/tmp/labm8_py.tmp")


# cd(), cdpop()
def test_cd():
    cwd = os.getcwd()
    new = fs.abspath("..")

    assert new == fs.cd("..")
    assert new == os.getcwd()

    assert cwd == fs.cdpop()
    assert cwd == os.getcwd()

    assert cwd == fs.cdpop()
    assert cwd == os.getcwd()

    assert cwd == fs.cdpop()
    assert cwd == os.getcwd()


# pwd()
def test_pwd():
    assert os.getcwd() == fs.pwd()


# exists()
def test_exists():
    assert fs.exists(__file__)
    assert fs.exists("/")
    assert not fs.exists("/not/a/real/path (I hope!)")


# isfile()
def test_isfile():
    assert fs.isfile(__file__)
    assert not fs.isfile("/")
    assert not fs.isfile("/not/a/real/path (I hope!)")


# isexe()
def test_isexe():
    assert fs.isexe("/bin/ls")
    assert not fs.isexe("/home")
    assert not fs.isexe("/not/a/real/path (I hope!)")


# isdir()
def test_isdir():
    assert not fs.isdir(__file__)
    assert fs.isdir("/")
    assert not fs.isdir("/not/a/real/path (I hope!)")


# read()
def test_read():
    assert ["Hello, world!"] == fs.read("labm8/py/test_data/hello_world")
    assert [
        "# data1 - test file",
        "This",
        "is a test file",
        "With",
        "trailing  # comment",
        "",
        "",
        "",
        "whitespace",
        "0.344",
    ] == fs.read("labm8/py/test_data/data1")


def test_read_no_rstrip():
    assert [
        "# data1 - test file\n",
        "This\n",
        "is a test file\n",
        "With\n",
        "trailing  # comment  \n",
        "\n",
        "\n",
        "\n",
        "whitespace\n",
        "0.344\n",
    ] == fs.read("labm8/py/test_data/data1", rstrip=False)


def test_read_ignore_comments():
    assert [
        "This",
        "is a test file",
        "With",
        "trailing",
        "",
        "",
        "",
        "whitespace",
        "0.344",
    ] == fs.read("labm8/py/test_data/data1", comment_char="#")


def test_read_ignore_comments_no_rstrip():
    assert [
        "This\n",
        "is a test file\n",
        "With\n",
        "trailing  ",
        "\n",
        "\n",
        "\n",
        "whitespace\n",
        "0.344\n",
    ] == fs.read("labm8/py/test_data/data1", rstrip=False, comment_char="#")


def test_read_empty_file():
    assert fs.read("labm8/py/test_data/empty_file") == []


# mkdir()
def test_mkdir():
    fs.rm("/tmp/labm8_py.dir")
    assert not fs.isdir("/tmp/labm8_py.dir")
    fs.mkdir("/tmp/labm8_py.dir")
    assert fs.isdir("/tmp/labm8_py.dir")


def test_mkdir_parents():
    assert not fs.isdir("/tmp/labm8_py.dir/foo/bar")
    fs.mkdir("/tmp/labm8_py.dir/foo/bar")
    assert fs.isdir("/tmp/labm8_py.dir/foo/bar")


def test_mkdir_exists():
    fs.mkdir("/tmp/labm8_py.dir/")
    assert fs.isdir("/tmp/labm8_py.dir/")
    fs.mkdir("/tmp/labm8_py.dir/")
    fs.mkdir("/tmp/labm8_py.dir/")
    assert fs.isdir("/tmp/labm8_py.dir/")


# mkopen()
def test_mkopen():
    fs.rm("/tmp/labm8_py.dir")
    assert not fs.isdir("/tmp/labm8_py.dir/")
    f = fs.mkopen("/tmp/labm8_py.dir/foo", "w")
    assert fs.isdir("/tmp/labm8_py.dir/")
    f.close()


# rm()
def test_rm():
    system.echo("Hello, world!", "/tmp/labm8_py.tmp")
    assert fs.isfile("/tmp/labm8_py.tmp")
    fs.rm("/tmp/labm8_py.tmp")
    assert not fs.isfile("/tmp/labm8_py.tmp")
    fs.rm("/tmp/labm8_py.tmp")
    fs.rm("/tmp/labm8_py.tmp")
    fs.rm("/tmp/labm8_py.dir")
    fs.mkdir("/tmp/labm8_py.dir/foo/bar")
    system.echo("Hello, world!", "/tmp/labm8_py.dir/foo/bar/baz")
    assert fs.isfile("/tmp/labm8_py.dir/foo/bar/baz")
    fs.rm("/tmp/labm8_py.dir")
    assert not fs.isfile("/tmp/labm8_py.dir/foo/bar/baz")
    assert not fs.isfile("/tmp/labm8_py.dir/")


def test_rm_glob():
    fs.mkdir("/tmp/labm8_py.glob")
    system.echo("Hello, world!", "/tmp/labm8_py.glob/1")
    system.echo("Hello, world!", "/tmp/labm8_py.glob/2")
    system.echo("Hello, world!", "/tmp/labm8_py.glob/abc")

    fs.rm("/tmp/labm8_py.glob/a*", glob=False)
    assert fs.isfile("/tmp/labm8_py.glob/1")
    assert fs.isfile("/tmp/labm8_py.glob/2")
    assert fs.isfile("/tmp/labm8_py.glob/abc")

    fs.rm("/tmp/labm8_py.glob/a*")
    assert fs.isfile("/tmp/labm8_py.glob/1")
    assert fs.isfile("/tmp/labm8_py.glob/2")
    assert not fs.isfile("/tmp/labm8_py.glob/abc")

    fs.rm("/tmp/labm8_py.glob/*")
    assert not fs.isfile("/tmp/labm8_py.glob/1")
    assert not fs.isfile("/tmp/labm8_py.glob/2")
    assert not fs.isfile("/tmp/labm8_py.glob/abc")


# rmtrash()
@test.Skip(
    reason="Insufficient access privileges for operation on macOS",
)
def test_rmtrash():
    with tempfile.NamedTemporaryFile(prefix="labm8_") as f:
        assert fs.isfile(f.name)
        fs.rmtrash(f.name)
        assert not fs.isfile(f.name)
        fs.rmtrash(f.name)
        fs.rm(f.name)
    with tempfile.TemporaryDirectory() as d:
        fs.rm(d)
        fs.mkdir(d, "foo/bar")
        system.echo("Hello, world!", fs.path(d, "foo/bar/baz"))
        assert fs.isfile(f, "foo/bar/baz")
        fs.rmtrash(d)
        assert not fs.isfile(d, "foo/bar/baz")
        assert not fs.isdir(d)


def test_rmtrash_bad_path():
    fs.rmtrash("/not/a/real/path")


# cp()
def test_cp():
    system.echo("Hello, world!", "/tmp/labm8_py.tmp")
    assert ["Hello, world!"] == fs.read("/tmp/labm8_py.tmp")
    # Cleanup any existing file.
    fs.rm("/tmp/labm8_py.tmp.copy")
    assert not fs.exists("/tmp/labm8_py.tmp.copy")
    fs.cp("/tmp/labm8_py.tmp", "/tmp/labm8_py.tmp.copy")
    assert fs.read("/tmp/labm8_py.tmp") == fs.read("/tmp/labm8_py.tmp.copy")


def test_cp_no_file():
    with test.Raises(IOError):
        fs.cp("/not a real src", "/not/a/real dest")


def test_cp_dir():
    fs.rm("/tmp/labm8_py")
    fs.rm("/tmp/labm8_py.copy")
    fs.mkdir("/tmp/labm8_py/foo/bar")
    assert not fs.exists("/tmp/labm8_py.copy")
    fs.cp("/tmp/labm8_py/", "/tmp/labm8_py.copy")
    assert fs.isdir("/tmp/labm8_py.copy")
    assert fs.isdir("/tmp/labm8_py.copy/foo")
    assert fs.isdir("/tmp/labm8_py.copy/foo/bar")


def test_cp_overwrite():
    system.echo("Hello, world!", "/tmp/labm8_py.tmp")
    assert ["Hello, world!"] == fs.read("/tmp/labm8_py.tmp")
    # Cleanup any existing file.
    fs.rm("/tmp/labm8_py.tmp.copy")
    assert not fs.exists("/tmp/labm8_py.tmp.copy")
    fs.cp("/tmp/labm8_py.tmp", "/tmp/labm8_py.tmp.copy")
    system.echo("Goodbye, world!", "/tmp/labm8_py.tmp")
    fs.cp("/tmp/labm8_py.tmp", "/tmp/labm8_py.tmp.copy")
    assert fs.read("/tmp/labm8_py.tmp") == fs.read("/tmp/labm8_py.tmp.copy")


def test_cp_over_dir():
    fs.mkdir("/tmp/labm8_py.tmp.src")
    system.echo("Hello, world!", "/tmp/labm8_py.tmp.src/foo")
    fs.rm("/tmp/labm8_py.tmp.copy")
    fs.mkdir("/tmp/labm8_py.tmp.copy")
    assert fs.isdir("/tmp/labm8_py.tmp.src")
    assert fs.isfile("/tmp/labm8_py.tmp.src/foo")
    assert fs.isdir("/tmp/labm8_py.tmp.copy")
    assert not fs.isfile("/tmp/labm8_py.tmp.copy/foo")
    fs.cp("/tmp/labm8_py.tmp.src", "/tmp/labm8_py.tmp.copy/")
    assert fs.isdir("/tmp/labm8_py.tmp.src")
    assert fs.isfile("/tmp/labm8_py.tmp.src/foo")
    assert fs.isdir("/tmp/labm8_py.tmp.copy")
    assert fs.isfile("/tmp/labm8_py.tmp.copy/foo")
    assert fs.read("/tmp/labm8_py.tmp.src/foo") == fs.read("/tmp/labm8_py.tmp.copy/foo")


# mv()
def test_mv():
    system.echo("Hello, world!", "/tmp/labm8_py.tmp")
    assert ["Hello, world!"] == fs.read("/tmp/labm8_py.tmp")
    # Cleanup any existing file.
    fs.rm("/tmp/labm8_py.tmp.copy")
    assert not fs.exists("/tmp/labm8_py.tmp.copy")
    fs.mv("/tmp/labm8_py.tmp", "/tmp/labm8_py.tmp.copy")
    assert ["Hello, world!"] == fs.read("/tmp/labm8_py.tmp.copy")
    assert not fs.exists("/tmp/labm8_py.tmp")


def test_mv_no_src():
    with test.Raises(fs.File404):
        fs.mv("/bad/path", "foo")


def test_mv_no_dst():
    system.echo("Hello, world!", "/tmp/labm8_py.tmp")
    with test.Raises(IOError):
        fs.mv("/tmp/labm8_py.tmp", "/not/a/real/path")
    fs.rm("/tmp/labm8_py.tmp")


# ls()
def test_ls():
    assert ["a", "b", "c", "d"] == fs.ls("labm8/py/test_data/testdir")


def test_ls_recursive():
    assert fs.ls("labm8/py/test_data/testdir", recursive=True) == [
        "a",
        "b",
        "c",
        "c/e",
        "c/f",
        "c/f/f",
        "c/f/f/i",
        "c/f/h",
        "c/g",
        "d",
    ]


def test_ls_abspaths():
    fs.cp("labm8/py/test_data/testdir", "/tmp/testdir")
    assert fs.ls("/tmp/testdir", abspaths=True) == [
        "/tmp/testdir/a",
        "/tmp/testdir/b",
        "/tmp/testdir/c",
        "/tmp/testdir/d",
    ]
    assert fs.ls("/tmp/testdir", recursive=True, abspaths=True) == [
        "/tmp/testdir/a",
        "/tmp/testdir/b",
        "/tmp/testdir/c",
        "/tmp/testdir/c/e",
        "/tmp/testdir/c/f",
        "/tmp/testdir/c/f/f",
        "/tmp/testdir/c/f/f/i",
        "/tmp/testdir/c/f/h",
        "/tmp/testdir/c/g",
        "/tmp/testdir/d",
    ]
    fs.rm("/tmp/testdir")


def test_ls_empty_dir():
    fs.mkdir("/tmp/labm8_py.empty")
    assert not fs.ls("/tmp/labm8_py.empty")
    fs.rm("/tmp/labm8_py.empty")


def test_ls_bad_path():
    with test.Raises(OSError):
        fs.ls("/not/a/real/path/bro")


def test_ls_single_file():
    assert ["a"] == fs.ls("labm8/py/test_data/testdir/a")


# lsdirs()
def test_lsdirs():
    assert ["c"] == fs.lsdirs("labm8/py/test_data/testdir")


def test_lsdirs_recursive():
    assert fs.lsdirs("labm8/py/test_data/testdir", recursive=True) == [
        "c",
        "c/f",
        "c/f/f",
    ]


def test_lsdirs_bad_path():
    with test.Raises(OSError):
        fs.lsdirs("/not/a/real/path/bro")


def test_lsdirs_single_file():
    assert not fs.lsdirs("labm8/py/test_data/testdir/a")


# lsdirs()
def test_lsfiles():
    assert fs.lsfiles("labm8/py/test_data/testdir") == ["a", "b", "d"]


def test_lsfiles_recursive():
    assert fs.lsfiles("labm8/py/test_data/testdir", recursive=True) == [
        "a",
        "b",
        "c/e",
        "c/f/f/i",
        "c/f/h",
        "c/g",
        "d",
    ]


def test_lsfiles_bad_path():
    with test.Raises(OSError):
        fs.lsfiles("/not/a/real/path/bro")


def test_lsfiles_single_file():
    assert fs.lsfiles("labm8/py/test_data/testdir/a") == ["a"]


def test_directory_is_empty_empty_dir():
    """Test that en empty directory returns True."""
    with tempfile.TemporaryDirectory() as d:
        assert fs.directory_is_empty(d)


def test_directory_is_empty_only_subdirs():
    """Test that a subdirectory means the directory is not empty."""
    with tempfile.TemporaryDirectory() as d:
        (pathlib.Path(d) / "a").mkdir()
        assert not fs.directory_is_empty(d)


def test_directory_is_empty_file():
    """Test that a file means the directory is not empty."""
    with tempfile.TemporaryDirectory() as d:
        (pathlib.Path(d) / "a").touch()
        assert not fs.directory_is_empty(d)


def test_directory_is_empty_non_existent():
    """Test that a non-existent path is an empty directory."""
    with tempfile.TemporaryDirectory() as d:
        assert fs.directory_is_empty(pathlib.Path(d) / "a")


def test_directory_is_empty_file_argument():
    """Test that path to a file is an empty directory."""
    with tempfile.TemporaryDirectory() as d:
        (pathlib.Path(d) / "a").touch()
        assert fs.directory_is_empty(pathlib.Path(d) / "a")


# chdir()


def test_chdir_yield_value():
    """Test that chdir() yields the requested directory as a pathlib.Path."""
    with tempfile.TemporaryDirectory() as d:
        with fs.chdir(d) as d2:
            assert pathlib.Path(d) == d2


def test_chdir_cwd():
    """Test that chdir() correctly changes the working directory."""
    with tempfile.TemporaryDirectory() as d:
        with fs.chdir(d):
            assert os.getcwd().endswith(d)


def test_chdir_not_a_directory():
    """Test that FileNotFoundError is raised if requested path does not exist."""
    with test.Raises(FileNotFoundError):
        with fs.chdir("/not/a/real/path"):
            pass


def test_chdir_file_argument():
    """Test that NotADirectoryError is raised if requested path is a file."""
    with tempfile.NamedTemporaryFile(prefix="labm8_") as f:
        with test.Raises(NotADirectoryError):
            with fs.chdir(f.name):
                pass


def test_TemporaryFileWithContents_contents():
    """Test that temporary file has expected contents."""
    contents = "Hello, world!"
    with fs.TemporaryFileWithContents(contents.encode("utf-8")) as f:
        filename = f.name
        contents_read = open(filename).read()
        assert contents_read == contents

    assert not os.path.exists(filename)


def test_Write_overwrite(tempdir: pathlib.Path):
    fs.Write(tempdir / "file.txt", "original contents".encode("utf-8"))
    fs.Write(tempdir / "file.txt", "Hello, world!".encode("utf-8"))
    with open(tempdir / "file.txt") as fp:
        assert fp.read() == "Hello, world!"


def test_Write_exclusive(tempdir: pathlib.Path):
    fs.Write(tempdir / "file.txt", "original contents".encode("utf-8"))
    with test.Raises(OSError):
        fs.Write(
            tempdir / "file.txt",
            "Hello, world!".encode("utf-8"),
            overwrite_existing=False,
        )


def test_Write_mode(tempdir: pathlib.Path):
    mode = 0o0744
    fs.Write(tempdir / "file.txt", "Hello, world!".encode("utf-8"), mode=mode)
    s = os.stat(str(tempdir / "file.txt"))
    assert stat.S_IMODE(s.st_mode) == mode


def test_Atomic_write_successful(tempdir: pathlib.Path):
    fs.AtomicWrite(tempdir / "file.txt", "Hello, world!".encode("utf-8"))
    with open(tempdir / "file.txt") as fp:
        assert fp.read() == "Hello, world!"


def test_Atomic_write_mode(tempdir: pathlib.Path):
    mode = 0o0745
    fs.AtomicWrite(
        tempdir / "file.txt",
        "Hello, world!".encode("utf-8"),
        mode=mode,
    )
    s = os.stat(str(tempdir / "file.txt"))
    assert stat.S_IMODE(s.st_mode) == mode


if __name__ == "__main__":
    test.Main()
