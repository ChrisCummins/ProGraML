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

The API is divided into three types of operations: graph *creation*, graph
*transformation*, and graph *serialization*, all available under the
:code:`programl` namespace.

ProGraML was first described in this `this paper
<https://chriscummins.cc/pub/2021-icml.pdf>`_:

    Cummins, C., Fisches, Z., Ben-Nun, T., Hoefler, T., O'Boyle, M., and
    Leather, H. "ProGraML: A Graph-based Program Representation for Data Flow
    Analysis and Compiler Optimizations." In 38th International Conference on
    Machine Learning (ICML).
"""
from pathlib import Path

from programl.create_ops import (
    CLANG_VERSIONS,
    LLVM_VERSIONS,
    from_clang,
    from_cpp,
    from_llvm_ir,
    from_xla_hlo_proto,
)
from programl.exceptions import (
    GraphCreationError,
    GraphTransformError,
    UnsupportedCompiler,
)
from programl.proto import ProgramGraph
from programl.serialize_ops import (
    from_bytes,
    from_string,
    load_graphs,
    save_graphs,
    to_bytes,
    to_string,
)
from programl.transform_ops import to_dgl, to_dot, to_json, to_networkx
from programl.util.py.runfiles_path import runfiles_path
from programl.version import PROGRAML_VERSION

__version__ = PROGRAML_VERSION
__author__ = "Chris Cummins"
__email__ = "chrisc.101@gmail.com"
__copyright__ = "Copyright 2019-2020 the ProGraML authors"
__license__ = "Apache License, Version 2.0"

binaries_path: Path = runfiles_path("programl/bin")

__all__ = [
    "binaries_path",
    "CLANG_VERSIONS",
    "from_bytes",
    "from_clang",
    "from_cpp",
    "from_llvm_ir",
    "from_string",
    "from_xla_hlo_proto",
    "GraphCreationError",
    "GraphTransformError",
    "LLVM_VERSIONS",
    "load_graphs",
    "ProgramGraph",
    "save_graphs",
    "to_bytes",
    "to_dgl",
    "to_dot",
    "to_json",
    "to_networkx",
    "to_string",
    "UnsupportedCompiler",
]
