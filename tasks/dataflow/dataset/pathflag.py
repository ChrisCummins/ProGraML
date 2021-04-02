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
"""This module defines a flag, --path. You can access it by calling path(). I
wrote it because a flag can only be defined once, and multiple files need
a path flag. It may be the greatest code I have written.

I hope you're having a nice day. :-)
"""
import pathlib

from absl import flags

flags.DEFINE_string(
    "path",
    str(pathlib.Path("~/programl/dataflow").expanduser()),
    "The dataset directory.",
)


def path():
    """Get --path value.

    In the form of a haiku:

      This is a flag path.
      You call it to return path.
      I am computer.
    """
    return flags.FLAGS.path
