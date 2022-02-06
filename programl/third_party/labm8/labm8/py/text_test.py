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
"""Unit tests for //labm8/py:text."""
import pytest
from labm8.py import app, test, text

FLAGS = app.FLAGS


# get_substring_idxs()
def test_get_substring_idxs():
    assert [0, 2] == text.get_substring_idxs("a", "aba")
    assert not text.get_substring_idxs("a", "bb")


# truncate()
def test_truncate():
    assert "foo" == text.truncate("foo", 100)
    assert "1234567890" == text.truncate("1234567890", 10)
    assert "12345..." == text.truncate("1234567890", 8)
    for i in range(10, 20):
        assert i == len(
            text.truncate(
                "The quick brown fox jumped " "over the slow lazy dog",
                i,
            ),
        )


def test_truncate_bad_maxchar():
    with test.Raises(text.TruncateError):
        text.truncate("foo", -1)
        text.truncate("foo", 3)


# distance()
def test_levenshtein():
    assert 0 == text.levenshtein("foo", "foo")
    assert 1 == text.levenshtein("foo", "fooo")
    assert 3 == text.levenshtein("foo", "")
    assert 1 == text.levenshtein("1234", "1 34")
    assert 1 == text.levenshtein("123", "1 3")


# diff()
def test_diff():
    assert 0 == text.diff("foo", "foo")
    assert 0.25 == text.diff("foo", "fooo")
    assert 1 == text.diff("foo", "")
    assert 0.25 == text.diff("1234", "1 34")
    assert (1 / 3) == text.diff("123", "1 3")


# Prefix tree operations.


def test_BuildPrefixTree_empty():
    """Test that an empty prefix tree has a single node (the root)."""
    trie = text.BuildPrefixTree(set())
    assert trie.number_of_nodes() == 1


def test_BuildPrefixTree_abc():
    """Test that a single-world prefix tree has expected graph."""
    trie = text.BuildPrefixTree(
        {
            "abc",
        }
    )
    assert trie.number_of_nodes() == 4
    assert trie.number_of_edges() == 3


def test_PrefixTreeWords_abc():
    """Test that a single-world prefix tree has expected graph."""
    trie = text.BuildPrefixTree({"abc"})
    assert text.PrefixTreeWords(trie) == {"abc"}


def test_PrefixTreeWords_subwords():
    """Test that a single-world prefix tree has expected graph."""
    trie = text.BuildPrefixTree({"abc", "abcd"})
    assert text.PrefixTreeWords(trie) == {"abc", "abcd"}


def test_AutoCompletePrefix_whole_string_match():
    """Test that autocomplete matches whole word."""
    trie = text.BuildPrefixTree({"abc"})
    assert text.AutoCompletePrefix("abc", trie) == {"abc"}


def test_AutoCompletePrefix_single_substring_match():
    """Test that autocomplete matches a single substring."""
    trie = text.BuildPrefixTree({"abc"})
    assert text.AutoCompletePrefix("a", trie) == {"abc"}


def test_AutoCompletePrefix_multiple_substring_match():
    """Test that autocomplete matches multiple substrings."""
    trie = text.BuildPrefixTree({"abc", "abcd"})
    assert text.AutoCompletePrefix("ab", trie) == {"abc", "abcd"}


def test_AutoCompletePrefix_not_wound():
    """Test that autocomplete raises error for non-matching substring."""
    trie = text.BuildPrefixTree({"abc"})
    with test.Raises(KeyError):
        text.AutoCompletePrefix("d", trie)


def test_StripSingleLineComments_empty_string():
    assert text.StripSingleLineComments("") == ""


def test_StripSingleLineComments_no_comments():
    assert text.StripSingleLineComments("Hello, world") == "Hello, world"


def test_StripSingleLineComments_one_line_string_with_bash_comment():
    assert (
        text.StripSingleLineComments("Hello, world  # This is a comment")
        == "Hello, world  "
    )


def test_StripSingleLineComments_one_line_string_with_c_comment():
    assert (
        text.StripSingleLineComments("Hello, world  // This is a comment")
        == "Hello, world  "
    )


def test_StripSingleLineComments_multiline_string_with_comments():
    assert (
        text.StripSingleLineComments(
            """
This
is#a comment  //
a  // multiline.
"""
        )
        == """
This
is
a  \n"""
    )


def test_StripSingleLineComments_custom_start_comment_re():
    assert (
        text.StripSingleLineComments(
            "This has an -- SQL comment",
            start_comment_re="--",
        )
        == "This has an "
    )


if __name__ == "__main__":
    test.Main()
