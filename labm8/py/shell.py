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
"""Utility code for working with shells."""
import os


class ShellEscapeCodes(object):
  """Shell escape codes for pretty-printing."""

  PURPLE = "\033[95m"
  CYAN = "\033[96m"
  DARKCYAN = "\033[36m"
  BLUE = "\033[94m"
  GREEN = "\033[92m"
  YELLOW = "\033[93m"
  RED = "\033[91m"
  BOLD = "\033[1m"
  UNDERLINE = "\033[4m"
  END = "\033[0m"


def ShellEscapeList(words):
  """Turn a list of words into a shell-safe string.

  Args:
    words: A list of words, e.g. for a command.

  Returns:
    A string of shell-quoted and space-separated words.
  """
  # Adapted from <https://github.com/google/google-apputils>.
  # Copyright 2007 Google Inc. All Rights Reserved.
  #
  # Licensed under the Apache License, Version 2.0 (the "License");
  # you may not use this file except in compliance with the License.
  # You may obtain a copy of the License at
  #
  #      http://www.apache.org/licenses/LICENSE-2.0
  #
  # Unless required by applicable law or agreed to in writing, software
  # distributed under the License is distributed on an "AS-IS" BASIS,
  # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  # See the License for the specific language governing permissions and
  # limitations under the License.
  if os.name == "nt":
    return " ".join(words)

  s = ""
  for word in words:
    # Single quote word, and replace each ' in word with '"'"'
    s += "'" + word.replace("'", "'\"'\"'") + "' "

  return s[:-1]
