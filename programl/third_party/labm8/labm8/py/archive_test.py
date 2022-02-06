"""Unit tests for //labm8/py:archive."""
import pathlib
import re
import tarfile
import zipfile

import pytest
from labm8.py import app, archive, test

FLAGS = app.FLAGS


@test.Fixture(
    scope="function",
    params=[
        # Parameterized by tuple:
        #   <file_extension>, <open_function>, <add_function>
        (".zip", lambda f: zipfile.ZipFile(f, "w"), lambda a: a.write),
        (".tar.bz2", lambda f: tarfile.open(f, "w:bz2"), lambda a: a.add),
    ],
    # Parameter tuple names.
    names=["zip", "tar.bz2"],
)
def test_archive(request, tempdir: pathlib.Path) -> pathlib.Path:
    """Yield path to an archive containing a single 'a.txt' file."""
    extension, open_fn, write_fn = request.param
    path = tempdir / f"a{extension}"
    with open(tempdir / "a.txt", "w") as f:
        f.write("Hello, world!")
    with open_fn(path) as a:
        write_fn(a)(tempdir / "a.txt", arcname="a.txt")
    (tempdir / "a.txt").unlink()
    yield path


def Touch(path: pathlib.Path) -> pathlib.Path:
    with open(path, "w") as f:
        pass
    return path


def test_Archive_path_not_found(tempdir: pathlib.Path):
    """Test that FileNotFound raised if path doesn't exist."""
    with test.Raises(FileNotFoundError) as e_ctx:
        archive.Archive(tempdir / "a.zip")
    assert str(e_ctx.value).startswith("No such file: '")


def test_Archive_no_suffix(tempdir: pathlib.Path):
    """Test that error raised if path has no suffix."""
    Touch(tempdir / "a")
    with test.Raises(archive.UnsupportedArchiveFormat) as e_ctx:
        archive.Archive(tempdir / "a")
    assert str(e_ctx.value) == "Archive 'a' has no extension"


def test_Archive_assume_filename_no_suffix(tempdir: pathlib.Path):
    """Test that error raised if assumed path has no suffix."""
    Touch(tempdir / "a.zip")
    with test.Raises(archive.UnsupportedArchiveFormat) as e_ctx:
        archive.Archive(tempdir / "a.zip", assume_filename="a")
    assert str(e_ctx.value) == "Archive 'a' has no extension"


@test.Parametrize("suffix", (".foo", ".tar.abc"))
def test_Archive_unsupported_suffixes(tempdir: pathlib.Path, suffix: str):
    path = tempdir / f"a{suffix}"
    Touch(path)

    with test.Raises(archive.UnsupportedArchiveFormat) as e_ctx:
        archive.Archive(path)
    assert re.match(
        f"Unsupported file extension '(.+)' for archive 'a{suffix}'",
        str(e_ctx.value),
    )


def test_Archive_as_context_manager(test_archive: pathlib.Path):
    """Test context manager for a single file zip."""
    # Open the archive and check the contents.
    with archive.Archive(test_archive) as d:
        assert (d / "a.txt").is_file()
        assert len(list(d.iterdir())) == 1
        with open(d / "a.txt") as f:
            assert f.read() == "Hello, world!"


def test_Archive_ExtractAll(test_archive: pathlib.Path, tempdir: pathlib.Path):
    """Test ExtractAll for a single file zip."""
    # Open the archive and check that it still exists.
    archive.Archive(test_archive).ExtractAll(tempdir)
    assert test_archive.is_file()

    # Check the archive contents.
    assert (tempdir / "a.txt").is_file()
    assert len(list(tempdir.iterdir())) == 2  # the zip file and a.txt
    with open(tempdir / "a.txt") as f:
        assert f.read() == "Hello, world!"


def test_Archive_ExtractAll_parents(
    test_archive: pathlib.Path,
    tempdir: pathlib.Path,
):
    """Test that ExtractAll creates necessary parent directories"""
    # Open the archive and check that it still exists.
    archive.Archive(test_archive).ExtractAll(tempdir / "foo/bar/car")
    assert test_archive.is_file()

    # Check the archive contents.
    assert (tempdir / "foo/bar/car/a.txt").is_file()
    assert len(list(tempdir.iterdir())) == 2  # the zip file and 'foo/'
    with open(tempdir / "foo/bar/car/a.txt") as f:
        assert f.read() == "Hello, world!"


if __name__ == "__main__":
    test.Main()
