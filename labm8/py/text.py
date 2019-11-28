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
"""Text utilities.
"""
from __future__ import division

import re
import typing

import networkx as nx


class Error(Exception):
  """
  Module-level error.
  """
  pass


class TruncateError(Error):
  """
  Thrown in case of truncation error.
  """
  pass


def get_substring_idxs(substr, string):
  """
  Return a list of indexes of substr. If substr not found, list is
  empty.

  Arguments:
      substr (str): Substring to match.
      string (str): String to match in.

  Returns:
      list of int: Start indices of substr.
  """
  return [match.start() for match in re.finditer(substr, string)]


def truncate(string, maxchar):
  """
  Truncate a string to a maximum number of characters.

  If the string is longer than maxchar, then remove excess
  characters and append an ellipses.

  Arguments:

      string (str): String to truncate.
      maxchar (int): Maximum length of string in characters. Must be >= 4.

  Returns:

      str: Of length <= maxchar.

  Raises:

      TruncateError: In case of an error.
  """
  if maxchar < 4:
    raise TruncateError('Maxchar must be > 3')

  if len(string) <= maxchar:
    return string
  else:
    return string[:maxchar - 3] + '...'


def levenshtein(s1, s2):
  """
  Return the Levenshtein distance between two strings.

  Implementation of Levenshtein distance, one of a family of edit
  distance metrics.

  Based on: https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python

  Examples:

      >>> text.levensthein("foo", "foo")
      0

      >>> text.levensthein("foo", "fooo")
      1

      >>> text.levensthein("foo", "")
      3

      >>> text.levensthein("1234", "1 34")
      1

  Arguments:

      s1 (str): Argument A.
      s2 (str): Argument B.

  Returns:

      int: Levenshtein distance between the two strings.
  """
  # Left string must be >= right string.
  if len(s1) < len(s2):
    return levenshtein(s2, s1)

  # Distance is length of s1 if s2 is empty.
  if len(s2) == 0:
    return len(s1)

  previous_row = range(len(s2) + 1)
  for i, c1 in enumerate(s1):
    current_row = [i + 1]
    for j, c2 in enumerate(s2):
      insertions = previous_row[j + 1] + 1
      deletions = current_row[j] + 1
      substitutions = previous_row[j] + (c1 != c2)
      current_row.append(min(insertions, deletions, substitutions))
    previous_row = current_row

  return previous_row[-1]


def diff(s1, s2):
  """
  Return a normalised Levenshtein distance between two strings.

  Distance is normalised by dividing the Levenshtein distance of the
  two strings by the max(len(s1), len(s2)).

  Examples:

      >>> text.diff("foo", "foo")
      0

      >>> text.diff("foo", "fooo")
      1

      >>> text.diff("foo", "")
      1

      >>> text.diff("1234", "1 34")
      1

  Arguments:

      s1 (str): Argument A.
      s2 (str): Argument B.

  Returns:

      float: Normalised distance between the two strings.
  """
  return levenshtein(s1, s2) / max(len(s1), len(s2))


def AddWordToPrefixTree(trie: nx.DiGraph, word: str) -> None:
  """Add the given word to a prefix tree.

  Args:
    trie: The prefix tree.
    word: The word to add to the prefix tree.
  """
  current_node = 0
  for char in word:
    for neighbour_id in trie[current_node]:
      if trie.nodes[neighbour_id]['char'] == char:
        current_node = neighbour_id
        break
    else:
      new_node_id = max(trie.nodes) + 1
      trie.add_node(new_node_id, char=char)
      trie.add_edge(current_node, new_node_id)
      current_node = new_node_id

  trie.nodes[current_node]['word'] = word


def BuildPrefixTree(words: typing.Set[str]):
  """Construct a prefix tree from the given words.

  A prefix tree can be used for providing autocomplete results for a prefix,
  see AutoCompletePrefix(). This implementation uses a networkx directed graph
  to represent the tree, with a 'char' attribute attached to nodes, and a 'word'
  attribute containing whole-word sequences on terminal nodes.

  This implementation favors clarity over efficiency.

  Args:
    words: The words to construct a prefix tree from.

  Returns:
    A prefix tree.
  """
  trie = nx.DiGraph()
  trie.add_node(0)
  for word in words:
    AddWordToPrefixTree(trie, word)
  return trie


def PrefixTreeWords(trie: nx.DiGraph, root_node: int = 0) -> typing.List[str]:
  """Return all words in prefix tree.

  Args:
    trie: A prefix tree, constructed using BuildPrefixTree().
    root_node: The root node to get words from.

  Returns:
    The set of words in the prefix tree.
  """
  ret = set()
  word = trie.nodes[root_node].get('word')
  if word:
    ret.add(word)
  for neighbor_node in trie[root_node]:
    ret = ret.union(PrefixTreeWords(trie, root_node=neighbor_node))
  return ret


def AutoCompletePrefix(prefix: str, trie: nx.DiGraph) -> typing.Set[str]:
  """Return all words in prefix tree which start with the given prefix.

  Args:
    prefix: The prefix to match.
    trie: A prefix tree, constructed using BuildPrefixTree().

  Returns:
    The set of words in the prefix tree that begin with the prefix.

  Raises:
    ValueError: If prefix is not given.
    KeyError: If the prefix is not found.
  """
  if not prefix:
    raise ValueError('Prefix cannot be empty')

  ret = set()

  current_node = 0
  for char in prefix:
    for neighbour_id in trie[current_node]:
      if trie.nodes[neighbour_id]['char'] == char:
        current_node = neighbour_id
        node_word = trie.nodes[current_node].get('word')
        if node_word:
          ret.add(node_word)
        break
    else:
      raise KeyError(f"Prefix not found: '{prefix}'")

  return ret.union(PrefixTreeWords(trie, root_node=current_node))


def CamelCapsToUnderscoreSeparated(camel_caps_str: str):
  components = re.findall('[A-Z][^A-Z]*', camel_caps_str)
  assert components
  return '_'.join(x.lower() for x in components)


def StripSingleLineComments(
    string: str,
    start_comment_re: str = '(#|//)',
) -> str:
  """Strip line comments from a string.

  Args:
    string: The string to strip the comments of.
    start_comment_re: The regular expression to match the start of a line
      comment. By default, this matches Bash-style '#' and C-style '//'
      comments.

  Returns:
    The string.
  """
  comment_re = re.compile(f'{start_comment_re}.*')
  lines = [comment_re.sub('', line) for line in string.split('\n')]
  return '\n'.join(lines)
