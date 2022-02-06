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
"""Unit tests for //labm8/hashcache.py."""
import pathlib
import tempfile
import time

from labm8.py import app, hashcache, test

FLAGS = app.FLAGS

# The hash_fn arguments accepted by HashCache constructor.
HASH_FUNCTIONS = ["md5", "sha1", "sha256"]

# The known hashes of empty file.
EMPTY_FILE_HASHES = {
    "md5": "d41d8cd98f00b204e9800998ecf8427e",
    "sha1": "da39a3ee5e6b4b0d3255bfef95601890afd80709",
    "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
}


@test.Fixture(scope="function")
def database_path() -> pathlib.Path:
    """A test fixture which returns a path to use as a HashCache database."""
    with tempfile.TemporaryDirectory(prefix="labm8_hashcache_") as d:
        yield pathlib.Path(d) / "hashcache.db"


@test.Parametrize("hash_fn", HASH_FUNCTIONS)
def test_HashCache_unrecognized_hash_fn(database_path, hash_fn):
    """Test that a non-existent path raises an error."""
    with test.Raises(ValueError) as e_info:
        hashcache.HashCache(database_path, "null")
    assert "Hash function not recognized: 'null'" == str(e_info.value)


@test.Parametrize("hash_fn", HASH_FUNCTIONS)
def test_HashCache_GetHash_non_existent_path(database_path, hash_fn):
    """Test that a non-existent path raises an error."""
    c = hashcache.HashCache(database_path, hash_fn)
    with tempfile.TemporaryDirectory() as d:
        with test.Raises(FileNotFoundError) as e_info:
            c.GetHash(pathlib.Path(d) / "a")
        assert f"File not found: '{d}/a'" == str(e_info.value)


@test.Parametrize("hash_fn", HASH_FUNCTIONS)
def test_HashCache_GetHash_empty_file_md5(database_path, hash_fn):
    """Test the hash of an empty file."""
    c = hashcache.HashCache(database_path, hash_fn)
    with tempfile.TemporaryDirectory() as d:
        (pathlib.Path(d) / "a").touch()
        assert EMPTY_FILE_HASHES[hash_fn] == c.GetHash(pathlib.Path(d) / "a")
        # Once more to test a cache hit.
        assert EMPTY_FILE_HASHES[hash_fn] == c.GetHash(pathlib.Path(d) / "a")


@test.Parametrize("hash_fn", HASH_FUNCTIONS)
def test_HashCache_GetHash_empty_directory(database_path, hash_fn):
    """Test the hash of an empty directory."""
    c = hashcache.HashCache(database_path, hash_fn)
    with tempfile.TemporaryDirectory() as d:
        assert EMPTY_FILE_HASHES[hash_fn] == c.GetHash(pathlib.Path(d))
        # Once more to test a cache hit.
        assert EMPTY_FILE_HASHES[hash_fn] == c.GetHash(pathlib.Path(d))


@test.Parametrize("hash_fn", HASH_FUNCTIONS)
def test_HashCache_GetHash_unmodified_directory(database_path, hash_fn):
    """Test that an unmodified file returns the same hash."""
    c = hashcache.HashCache(database_path, hash_fn)
    with tempfile.TemporaryDirectory() as d:
        hash_1 = c.GetHash(pathlib.Path(d))
        hash_2 = c.GetHash(pathlib.Path(d))
        assert hash_1 == hash_2


@test.XFail(reason="Fix me")
@test.Parametrize("hash_fn", HASH_FUNCTIONS)
def test_HashCache_GetHash_modified_directory(database_path, hash_fn):
    """Test that modifying a directory changes the hash."""
    c = hashcache.HashCache(database_path, hash_fn)
    with tempfile.TemporaryDirectory() as d:
        hash_1 = c.GetHash(pathlib.Path(d))
        time.sleep(1)
        (pathlib.Path(d) / "a").touch()
        (pathlib.Path(d) / "b").touch()
        hash_2 = c.GetHash(pathlib.Path(d))
        assert hash_1 != hash_2


@test.Parametrize("hash_fn", HASH_FUNCTIONS)
def test_HashCache_GetHash_unmodified_file(database_path, hash_fn):
    """Test that an unmodified file returns the same hash."""
    c = hashcache.HashCache(database_path, hash_fn)
    with tempfile.TemporaryDirectory() as d:
        (pathlib.Path(d) / "a").touch()
        hash_1 = c.GetHash(pathlib.Path(d) / "a")
        # Touch does not change the contents of the file, but will cause a
        # cache miss because of the changed mtime timestamp.
        (pathlib.Path(d) / "a").touch()
        hash_2 = c.GetHash(pathlib.Path(d) / "a")
        assert hash_1 == hash_2


@test.Parametrize("hash_fn", HASH_FUNCTIONS)
def test_HashCache_GetHash_in_memory_modified_file(database_path, hash_fn):
    """Test that modifying a file does not change the hash if in memory.

    This test emphasizes the danger of the in-memory hash, as it means that the
    validity of the cache is tied to the lifecycle of the process.
    """
    c = hashcache.HashCache(database_path, hash_fn, keep_in_memory=True)
    with tempfile.TemporaryDirectory() as d:
        (pathlib.Path(d) / "a").touch()
        hash_1 = c.GetHash(pathlib.Path(d) / "a")
        time.sleep(1)
        with open(pathlib.Path(d) / "a", "w") as f:
            f.write("Hello")
        hash_2 = c.GetHash(pathlib.Path(d) / "a")
        assert hash_1 == hash_2
        # Clear the in-memory cache and re-run the test. Now it will be a cache miss
        # and the correct hash will be returned.
        c.Clear()
        hash_3 = c.GetHash(pathlib.Path(d) / "a")
        assert hash_1 != hash_3


@test.Parametrize("hash_fn", HASH_FUNCTIONS)
def test_HashCache_GetHash_modified_file(database_path, hash_fn):
    """Test that modifying a file changes the hash."""
    c = hashcache.HashCache(database_path, hash_fn)
    with tempfile.TemporaryDirectory() as d:
        (pathlib.Path(d) / "a").touch()
        hash_1 = c.GetHash(pathlib.Path(d) / "a")
        time.sleep(1)
        with open(pathlib.Path(d) / "a", "w") as f:
            f.write("Hello")
        hash_2 = c.GetHash(pathlib.Path(d) / "a")
        assert hash_1 != hash_2


if __name__ == "__main__":
    test.Main()
