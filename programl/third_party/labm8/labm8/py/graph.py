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
"""This module defines classes and helper functions for working with graphs."""
import typing

from labm8.py import app, fmt

FLAGS = app.FLAGS


class Graph(object):
    def __init__(
        self,
        name: str,
        children: typing.Optional[typing.List["Graph"]] = None,
    ):
        self.name = name
        if children:
            self.children = set(children)
        else:
            self.children = set()

    def _ToDot(self, strings, visited: typing.Set["Graph"]) -> None:
        if self in visited:
            return
        visited.add(self)
        strings.append(f'{self.name}[label="{self.name}"]')
        for child in self.children:
            strings.append(f"{self.name} -> {child.name}")
        for child in self.children:
            child._ToDot(strings, visited)

    def ToDot(self) -> str:
        strings = []
        self._ToDot(strings, set())
        dot = "\n".join(fmt.IndentList(2, strings))
        return f"digraph graphname {{\n{dot}\n}}"

    def _PreOrderApply(
        self,
        callback: typing.Callable[["Graph"], None],
        visited: typing.Set["Graph"],
    ) -> None:
        if self in visited:
            return
        visited.add(self)
        for child in self.children:
            child._PreOrderApply(callback, visited)

    def PreOrderApply(self, callback: typing.Callable[["Graph"], None]) -> None:
        """Apply the given callback to all nodes in the graph.

        Args:
          callback: A callback function whose sole arguments accepts a Graph
            instance.
        """
        self._PreOrderApply(callback, set())

    def __eq__(self, other):
        return self.name == other.name

    def __neq__(self, other):
        return not self == other

    def __lt__(self, other):
        return self.name < other.name

    def __le__(self, other):
        return self < other or self == other

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return self.name
