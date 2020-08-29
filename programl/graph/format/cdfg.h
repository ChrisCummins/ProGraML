// This file defines a converter from ProGraML graph representation to CDFG.
// The CDFG representation is described in:
//
//     Brauckmann, A., Ertel, S., Goens, A., & Castrillon, J. (2020).
//     Compiler-Based Graph Representations for Deep Learning Models of Code.
//     CC. https://dl.acm.org/doi/pdf/10.1145/3377555.3377894
//
// Copyright 2019-2020 the ProGraML authors.
//
// Contact Chris Cummins <chrisc.101@gmail.com>.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once

#include <vector>

#include "absl/container/flat_hash_map.h"
#include "programl/proto/program_graph.pb.h"

namespace programl {
namespace graph {
namespace format {

class CDFGBuilder {
 public:
  CDFGBuilder(){};

  // Construct a CDFG from the given ProGraML graph.
  [[nodiscard]] ProgramGraph Build(const ProgramGraph& graph);

  // Return a list of node indices used to create the CDFG. For example, given a
  // mapping from input graph to CDFG graph nodes of:
  //
  //     {
  //      1: -
  //      2: 0
  //      3: -,
  //      4: 1,
  //      5: 2,
  //     }
  //
  // GetNodeList() will return [2, 4, 5]. The node list reset after each call to
  // Build().
  const std::vector<int>& GetNodeList() const;

 protected:
  void Clear();

 private:
  // A list of old graph node indices in the order that they appear in the new
  // graph.
  std::vector<int> nodeList_;
};

// Convert a list of node indices into a map from node index to new node index.
// E.g. [1, 2, 5] returns a map {1: 0, 2: 1, 5: 2}.
absl::flat_hash_map<int, int> NodeListToTranslationMap(const std::vector<int>& nodeList);

}  // namespace format
}  // namespace graph
}  // namespace programl
