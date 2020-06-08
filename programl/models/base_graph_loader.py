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
"""This module defines the interface for graph loaders."""
from typing import Any
from typing import Iterable


class BaseGraphLoader(object):
  """Base class for loading graphs from some dataset.

  This class behaves like an iterator over tuples of program graph
  data, with the addition of a Stop() method to signal that no further
  tuples will be consumed. The type of iterable values is described by
  the class itself, accessible using loader.IterableType() method.
  This enables you to "swap" in and out different loaders with different
  behaviour.

  Example usage:

      graph_loader = MyGraphLoader(...)
      assert graph_loader.IterableType() == (
        ProgramGraph,
        ProgramGraphFeatures,
      )
      for graph, features in graph_loader:
        # ... do something with graphs
        if done:
          self.graph_loader.Stop()

  """

  def IterableType(self) -> Any:
    # TOOO: No subclass yet uses this iterable type!
    raise NotImplementedError("abstract class")

  def __iter__(self,) -> Iterable[IterableType]:
    raise NotImplementedError("abstract class")

  def Stop(self) -> None:
    raise NotImplementedError("abstract class")
