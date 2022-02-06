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
"""Unit tests for //labm8/py:cache."""
import tempfile

import pytest
from labm8.py import app, cache, fs, system, test

FLAGS = app.FLAGS


def _TestCacheOps(_cache):
    _cache.clear()

    # Item setter
    _cache["foo"] = 1
    _cache["bar"] = 2

    # in operator
    assert "foo" in _cache
    assert "bar" in _cache

    # Item getter
    assert 1 == _cache["foo"]
    assert 2 == _cache["bar"]

    # Lookup error
    assert "notakey" not in _cache
    with test.Raises(KeyError):
        _cache["notakey"]

    # get() method
    assert 1 == _cache.get("foo")
    assert 2 == _cache.get("bar")

    # get() method default
    assert not _cache.get("baz")
    assert 10 == _cache.get("baz", 10)

    # "del" operator
    del _cache["bar"]
    assert "baz" not in _cache

    _cache.clear()


def test_cache():
    # Test abstract interface.
    c = cache.Cache()
    with test.Raises(NotImplementedError):
        c.get("foo")
    with test.Raises(NotImplementedError):
        c.clear()
    with test.Raises(NotImplementedError):
        c.items()
    with test.Raises(NotImplementedError):
        c["foo"]
    with test.Raises(NotImplementedError):
        c["foo"] = 1
    with test.Raises(NotImplementedError):
        "foo" in c
    with test.Raises(NotImplementedError):
        del c["foo"]


def test_TransientCache():
    _cache = cache.TransientCache()
    _TestCacheOps(_cache)

    # Test copy constructor.
    _cache["foo"] = 1
    _cache["bar"] = 2
    _cache["baz"] = 3

    cache2 = cache.TransientCache(_cache)
    assert 1 == cache2["foo"]
    assert 2 == cache2["bar"]
    assert 3 == cache2["baz"]

    assert 1 == _cache["foo"]
    assert 2 == _cache["bar"]
    assert 3 == _cache["baz"]


def test_JsonCache():
    with tempfile.NamedTemporaryFile(prefix="labm8_") as f:
        # Load test-set
        fs.cp("labm8/py/test_data/jsoncache.json", f.name)
        _cache = cache.JsonCache(f.name)

        assert "foo" in _cache
        assert 1 == _cache["foo"]
        _TestCacheOps(_cache)

        # Test copy constructor.
        _cache["foo"] = 1
        _cache["bar"] = 2
        _cache["baz"] = 3

        with tempfile.NamedTemporaryFile(prefix="labm8_") as f2:
            cache2 = cache.JsonCache(f2.name, _cache)
            assert 1 == cache2["foo"]
            assert 2 == cache2["bar"]
            assert 3 == cache2["baz"]
            assert 1 == _cache["foo"]
            assert 2 == _cache["bar"]
            assert 3 == _cache["baz"]
            _cache.clear()
            # Set for next time.
            _cache["foo"] = 1
            _cache.write()


def test_FSCache_init_and_empty():
    c = cache.FSCache("/tmp/labm8_py-cache-init-and-empty")
    assert fs.isdir("/tmp/labm8_py-cache-init-and-empty")
    c.clear()
    assert not fs.isdir("/tmp/labm8_py-cache-init-and-empty")


def test_set_and_get():
    fs.rm("/tmp/labm8_py-cache-set-and-get")
    c = cache.FSCache("/tmp/labm8_py-cache-set-and-get")
    # create file
    system.echo("Hello, world!", "/tmp/labm8_py.testfile.txt")
    # sanity check
    assert fs.read("/tmp/labm8_py.testfile.txt") == ["Hello, world!"]
    # insert file into cache
    c["foobar"] = "/tmp/labm8_py.testfile.txt"
    # file must be in cache
    assert fs.isfile(c.keypath("foobar"))
    # file must have been moved
    assert not fs.isfile("/tmp/labm8_py.testfile.txt")
    # check file contents
    assert fs.read(c["foobar"]) == ["Hello, world!"]
    assert fs.read(c["foobar"]) == fs.read(c.get("foobar"))
    c.clear()


def test_FSCache_404():
    c = cache.FSCache("/tmp/labm8_py-cache-404")
    with test.Raises(KeyError):
        c["foobar"]
    with test.Raises(KeyError):
        del c["foobar"]
    assert not c.get("foobar")
    assert c.get("foobar", 5) == 5
    c.clear()


def test_FSCache_remove():
    c = cache.FSCache("/tmp/labm8_py-cache-remove")
    # create file
    system.echo("Hello, world!", "/tmp/labm8_py.test.remove.txt")
    # sanity check
    assert fs.read("/tmp/labm8_py.test.remove.txt") == ["Hello, world!"]
    # insert file into cache
    c["foobar"] = "/tmp/labm8_py.test.remove.txt"
    # sanity check
    assert fs.read(c["foobar"]) == ["Hello, world!"]
    # remove from cache
    del c["foobar"]
    with test.Raises(KeyError):
        c["foobar"]
    assert not c.get("foobar")
    c.clear()


def test_FSCache_dict_key():
    c = cache.FSCache("/tmp/labm8_py-cache-dict")
    # create file
    system.echo("Hello, world!", "/tmp/labm8_py.test.remove.txt")
    # sanity check
    assert fs.read("/tmp/labm8_py.test.remove.txt") == ["Hello, world!"]
    # insert file into cache
    key = {"a": 5, "c": [1, 2, 3]}
    c[key] = "/tmp/labm8_py.test.remove.txt"
    # check file contents
    assert fs.read(c[key]) == ["Hello, world!"]
    c.clear()


def test_FSCache_missing_key():
    c = cache.FSCache("/tmp/labm8_py-missing-key")
    with test.Raises(ValueError):
        c["foo"] = "/not/a/real/path"
    c.clear()


def test_FSCache_iter_len():
    c = cache.FSCache("/tmp/labm8_py-fscache-iter", escape_key=cache.escape_path)
    c.clear()
    system.echo("Hello, world!", "/tmp/labm8_py.testfile.txt")
    c["foo"] = "/tmp/labm8_py.testfile.txt"
    for path in c:
        assert path == c.keypath("foo")
    system.echo("Hello, world!", "/tmp/labm8_py.testfile.txt")
    c["bar"] = "/tmp/labm8_py.testfile.txt"
    assert len(c) == 2
    assert len(c.ls()) == 2
    assert "bar" in c.ls()
    assert "foo" in c.ls()
    c.clear()


if __name__ == "__main__":
    test.Main()
