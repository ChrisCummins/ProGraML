# Copyright 2019-2020 the ProGraML authors.
#
# Contact Chris Cummins <chrisc.101@gmail.com>.
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
"""Unit tests for //program/graph/format/py:nx_format."""
import networkx as nx

from programl.graph.format.py import nx_format
from programl.proto import ProgramGraph
from tests.test_main import main


def test_ProgramGraphToNetworkX_empty_graph():
    """Build from an empty proto."""
    proto = ProgramGraph()
    g = nx_format.ProgramGraphToNetworkX(proto)
    assert isinstance(g, nx.MultiDiGraph)
    assert not g.number_of_nodes()
    assert not g.number_of_edges()


if __name__ == "__main__":
    main()
