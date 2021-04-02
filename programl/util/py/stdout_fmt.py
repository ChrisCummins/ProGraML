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
"""This module defines a --stdout_fmt flag and method for writing stdout."""
import sys

import google.protobuf.json_format
from absl import app, flags

flags.DEFINE_string(
    "stdout_fmt",
    "pbtxt",
    "The format of output. Valid options are: "
    '"pbtxt" for a text-format protocol buffer, '
    '"pb" for a binary format protocol buffer, '
    'or "json" for JSON. '
    "Text format protocol buffers are recommended for human-readable output, "
    "binary-format for efficient and fast file storage, and JSON for "
    "processing "
    "with third-party tools such as `jq`.",
)
FLAGS = flags.FLAGS


def WriteStdout(proto) -> None:
    """Write the given protocol buffer to stdout.

    The format is determined by the --stdout_fmt flag.

    Args:
      proto: The protocol buffer instance to write to stdout.
    """
    if FLAGS.stdout_fmt == "pb":
        sys.stdout.buffer.write(proto.SerializeToString())
    elif FLAGS.stdout_fmt == "pbtxt":
        print(proto)
    elif FLAGS.stdout_fmt == "json":
        print(
            google.protobuf.json_format.MessageToJson(
                proto,
                preserving_proto_field_name=True,
            )
        )
    else:
        raise app.UsageError(
            f"Unknown --stdout_fmt={FLAGS.stdout_fmt}. "
            "Supported formats: pb,pbtxt,json"
        )
