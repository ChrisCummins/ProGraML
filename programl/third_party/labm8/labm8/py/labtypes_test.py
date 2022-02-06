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
"""Unit tests for //labm8/py:fmt."""
import pytest
from labm8.py import app, labtypes, test

FLAGS = app.FLAGS


# is_str()
def test_is_str():
    assert labtypes.is_str("Hello, World!")
    assert labtypes.is_str(str("Hello, World!"))
    assert not labtypes.is_str("Hello, World!".encode("utf-8"))
    assert not labtypes.is_str(bytes("Hello, World!".encode("utf-8")))
    assert not labtypes.is_str(8)
    assert not labtypes.is_str(["a", "b", "c"])
    assert not labtypes.is_str({"a": "b", "c": 18})


def test_is_str_seq():
    assert not labtypes.is_str(tuple([1]))
    assert not labtypes.is_str((1, 2))
    assert not labtypes.is_str([1])
    assert not labtypes.is_str([1, 2])


def test_is_str_num():
    assert not labtypes.is_str(1)
    assert not labtypes.is_str(1.3)


def test_is_str_dict():
    assert not labtypes.is_str({"foo": 100})
    assert not labtypes.is_str({10: ["a", "b", "c"]})


# is_dict() tests
def test_is_dict():
    assert labtypes.is_dict({"foo": 100})
    assert labtypes.is_dict({10: ["a", "b", "c"]})


def test_is_dict_str():
    assert not labtypes.is_dict("a")
    assert not labtypes.is_dict("abc")
    assert not labtypes.is_dict(["abc", "def"][0])


def test_is_dict_seq():
    assert not labtypes.is_dict(tuple([1]))
    assert not labtypes.is_dict((1, 2))
    assert not labtypes.is_dict([1])
    assert not labtypes.is_dict([1, 2])


def test_is_dict_num():
    assert not labtypes.is_dict(1)
    assert not labtypes.is_dict(1.3)


# is_seq() tests
def test_is_seq():
    assert labtypes.is_seq(tuple([1]))
    assert labtypes.is_seq((1, 2))
    assert labtypes.is_seq([1])
    assert labtypes.is_seq([1, 2])


def test_is_seq_str():
    assert not labtypes.is_seq("a")
    assert not labtypes.is_seq("abc")
    assert not labtypes.is_seq(["abc", "def"][0])


def test_is_seq_num():
    assert not labtypes.is_seq(1)
    assert not labtypes.is_seq(1.3)


def test_is_seq_dict():
    assert not labtypes.is_seq({"foo": 100})
    assert not labtypes.is_seq({10: ["a", "b", "c"]})


# flatten()
def test_flatten():
    assert labtypes.flatten([[1], [2, 3]]) == [1, 2, 3]


# PairwiseIterator()


def test_PairwiseIterator_empty_list():
    """Test that empty list produces no output."""
    assert list(labtypes.PairwiseIterator([])) == []


def test_PairwiseIterator_input_is_list():
    """Test when input is list."""
    generator = labtypes.PairwiseIterator([0, 1, 2, 3])
    assert next(generator) == (0, 1)
    assert next(generator) == (1, 2)
    assert next(generator) == (2, 3)
    with test.Raises(StopIteration):
        next(generator)


def test_PairwiseIterator_input_is_iterator():
    """Test when input is iterator."""
    generator = labtypes.PairwiseIterator(range(4))
    assert next(generator) == (0, 1)
    assert next(generator) == (1, 2)
    assert next(generator) == (2, 3)


def test_PairwiseIterator_input_is_string():
    """Test when input is list."""
    generator = labtypes.PairwiseIterator("hello")
    assert next(generator) == ("h", "e")
    assert next(generator) == ("e", "l")
    assert next(generator) == ("l", "l")
    assert next(generator) == ("l", "o")


# SetDiff()


def test_SetDiff_empty_inputs():
    """Test when inputs are empty."""
    assert labtypes.SetDiff([], []) == set()


def test_SetDiff_one_input_is_empty():
    """Test when one input is empty."""
    assert labtypes.SetDiff([1, 2, 3], []) == {1, 2, 3}
    assert labtypes.SetDiff([], [1, 2, 3]) == {1, 2, 3}


def test_SetDiff_matching_inputs():
    """Test when both inputs are the same."""
    assert labtypes.SetDiff([1, 2, 3], [1, 2, 3]) == set()


def test_SetDiff_overlapping_inputs():
    """Test when inputs overlap."""
    assert labtypes.SetDiff([1, 2], [1, 2, 3]) == {3}


def test_SetDiff_unmatching_types():
    """Test when inputs are of different types."""
    assert labtypes.SetDiff([1, 2, 3], ["a", "b"]) == {1, 2, 3, "a", "b"}


def test_SetDiff_input_ranges():
    """Test when inputs are iterators."""
    assert labtypes.SetDiff(range(3), range(4)) == {3}


# AllSubclassesOfClass()


class A(object):
    """Class for AllSubclassesOfClass() tests."""

    pass


def test_AllSubclassesOfClass_no_subclasses():
    """Test that class with no subclasses returns empty set."""
    assert not labtypes.AllSubclassesOfClass(A)


class B(object):
    """Class for AllSubclassesOfClass() tests."""

    pass


class C(B):
    """Class for AllSubclassesOfClass() tests."""

    pass


class D(B):
    """Class for AllSubclassesOfClass() tests."""

    pass


def test_AllSubclassesOfClass_direct_subclasses():
    """Test that direct subclasses are returned."""
    assert labtypes.AllSubclassesOfClass(B) == {C, D}


class E(object):
    """Class for AllSubclassesOfClass() tests."""

    pass


class F(E):
    """Class for AllSubclassesOfClass() tests."""

    pass


class G(F):
    """Class for AllSubclassesOfClass() tests."""

    pass


def test_AllSubclassesOfClass_direct_subclasses():
    """Test that direct subclasses are returned."""
    assert labtypes.AllSubclassesOfClass(E) == {F, G}


def test_Chunkify_empty_sequence():
    """Test that chunkify of an empty sequence returns an empty list."""
    assert list(labtypes.Chunkify([], 1)) == []


def test_Chunkify_123():
    """Test chunkify with single element returns."""
    assert list(labtypes.Chunkify([1, 2, 3], 1)) == [[1], [2], [3]]


def test_Chunkify_123_chunk_size_two():
    """Test chunkify with single element returns."""
    assert list(labtypes.Chunkify([1, 2, 3], 2)) == [[1, 2], [3]]


if __name__ == "__main__":
    test.Main()
