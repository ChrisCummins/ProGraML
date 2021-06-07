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
"""ProGraML is a graph-based program representation for data flow analysis and
compiler optimizations.
"""
from programl.create_ops import from_clang, from_cpp, from_llvm_ir, from_xla_hlo_proto
from programl.proto import ProgramGraph
from programl.serialization_ops import from_string, load_graphs, save_graphs, to_string
from programl.transform_ops import to_dgl, to_json, to_networkx
from programl.version import PROGRAML_VERSION

__version__ = PROGRAML_VERSION
__author__ = "Chris Cummins"
__email__ = "chrisc.101@gmail.com"
__copyright__ = "Copyright 2019-2020 the ProGraML authors"
__license__ = "Apache License, Version 2.0"

__all__ = [
    "to_json",
    "to_networkx",
    "to_dgl",
    "to_string",
    "from_string",
    "save_graphs",
    "load_graphs",
    "from_llvm_ir",
    "from_clang",
    "from_cpp",
    "from_xla_hlo_proto",
    "ProgramGraph",
]
