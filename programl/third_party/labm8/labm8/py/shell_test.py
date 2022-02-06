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
"""Unit tests for //labm8/py:shell."""
from labm8.py import shell, test


def test_EscapeList():
    # Empty list
    words = []
    assert shell.ShellEscapeList(words) == ""

    # Empty string
    words = [""]
    assert shell.ShellEscapeList(words) == "''"

    # Single word
    words = ["foo"]
    assert shell.ShellEscapeList(words) == "'foo'"

    # Single word with single quote
    words = ["foo'bar"]
    expected = """   'foo'"'"'bar'   """.strip()
    assert shell.ShellEscapeList(words) == expected
    # .. double quote
    words = ['foo"bar']
    expected = """   'foo"bar'   """.strip()
    assert shell.ShellEscapeList(words) == expected

    # Multiple words
    words = ["foo", "bar"]
    assert shell.ShellEscapeList(words) == "'foo' 'bar'"

    # Words with spaces
    words = ["foo", "bar", "foo'' ''bar"]
    expected = """   'foo' 'bar' 'foo'"'"''"'"' '"'"''"'"'bar'   """.strip()
    assert shell.ShellEscapeList(words) == expected

    # Now I'm just being mean
    words = ["foo", "bar", """   ""'"'"   """.strip()]
    expected = """   'foo' 'bar' '""'"'"'"'"'"'"'   """.strip()
    assert shell.ShellEscapeList(words) == expected


if __name__ == "__main__":
    test.Main()
