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
"""Encode node embeddings using inst2vec.

This program reads a single program graph from file, encodes it, and writes
the modified graph to stdout.

Example usage:

  Encode a program graph proto and write the result to file:

    $ inst2vec --ir=/tmp/source.ll < program.pbtxt > inst2vec.pbtxt
"""
from absl import app, flags

from programl.ir.llvm import inst2vec_encoder
from programl.proto import ProgramGraph
from programl.util.py.init_app import init_app
from programl.util.py.stdin_fmt import ParseStdinOrDie
from programl.util.py.stdout_fmt import WriteStdout

flags.DEFINE_string(
    "ir",
    None,
    "The path of the IR file that was used to construct the graph. This is "
    "required to inline struct definitions. This argument may be omitted when "
    "struct definitions do not need to be inlined.",
)
flags.DEFINE_string(
    "dataset",
    None,
    "The path of a directory to process. When set, this changes the behavior to "
    "instead iterate over all *.PrographGraph.pb protocol buffer files in the "
    "given directory, and adding inst2vec labels in-place. For each "
    "ProgramGraph.pb file, if a corresponding .ll file is found, that is used as "
    "the auxiliary IR file for inlining struct definitions.",
)
flags.DEFINE_string(
    "directory",
    None,
    "The path of a directory to process. When set, this changes the behavior to "
    "instead iterate over all *.PrographGraph.pb protocol buffer files in the "
    "given directory, and adding inst2vec labels in-place. For each "
    "ProgramGraph.pb file, if a corresponding .ll file is found, that is used as "
    "the auxiliary IR file for inlining struct definitions.",
)
FLAGS = flags.FLAGS


def main(argv):
    init_app(argv)
    encoder = inst2vec_encoder.Inst2vecEncoder()

    proto = ParseStdinOrDie(ProgramGraph())
    if FLAGS.ir:
        with open(FLAGS.ir) as f:
            ir = f.read()
    else:
        ir = None
    encoder.Encode(proto, ir)
    WriteStdout(proto)


if __name__ == "__main__":
    app.run(main)
