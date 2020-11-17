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
"""This module defines a --stdin_fmt flag and method for parsing stdin."""
import sys

import google.protobuf.json_format
from absl import flags

from programl.util.py import pbutil

flags.DEFINE_string(
    "stdin_fmt",
    "pbtxt",
    "The type of input format to use. Valid options are: "
    '"pbtxt" which reads a text format protocol buffer, '
    '"pb" which reads a binary format protocol buffer, '
    'or "json" which reads a JSON format protocol buffer.',
)
FLAGS = flags.FLAGS


def ParseStdinOrDie(proto, exit_code: int = 4):
    """Read the entire stdin and parse as a protocol buffer of the given type.

    The format is determined by the --stdin_fmt flag.

    Args:
      proto: A protocol buffer instance to set the parsed input to.
      exit_code: The exit code if parsing fails.

    Returns:
      The proto argument.
    """
    try:
        if FLAGS.stdin_fmt == "pb":
            proto.ParseFromString(sys.stdin.buffer.read())
        elif FLAGS.stdin_fmt == "pbtxt":
            pbutil.FromString(sys.stdin.buffer.read().decode("utf-8"), proto)
        elif FLAGS.stdin_fmt == "json":
            google.protobuf.json_format.Parse(sys.stdin.buffer.read(), proto)
        else:
            print(
                f"Unknown --stdin_fmt={FLAGS.stdin_fmt}. "
                "Supported formats: pb,pbtxt,json",
                file=sys.stderr,
            )
            sys.exit(exit_code)
    except (pbutil.DecodeError, google.protobuf.json_format.ParseError) as e:
        print(f"Failed to parse stdin: {e}", file=sys.stderr)
        sys.exit(exit_code)
    return proto
