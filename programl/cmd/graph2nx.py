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
"""Convert a ProgramGraph to NetworkX."""
import pickle
import sys

from labm8.py import app
from programl.graph.format.py import nx_format
from programl.proto import program_graph_pb2
from programl.util.py.stdin_fmt import ParseStdinOrDie


FLAGS = app.FLAGS


def Main():
  proto = ParseStdinOrDie(program_graph_pb2.ProgramGraph())
  pickle.dump(nx_format.ProgramGraphToNetworkX(proto), sys.stdout.buffer)


if __name__ == "__main__":
  app.Run(Main)
