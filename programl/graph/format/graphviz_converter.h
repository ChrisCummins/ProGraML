// This file defines a function for converting program graphs to graphviz.
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

#include "labm8/cpp/status.h"
#include "labm8/cpp/statusor.h"
#include "labm8/cpp/string.h"
#include "programl/proto/program_graph.pb.h"

namespace programl {
namespace graph {
namespace format {

// An enum describing the different node attributes that can be used for setting
// text labels.
enum NodeLabel {
  kNone = 0,
  kText,
  kFeature,
};

// Get the named node feature or return an error status if not found.
labm8::StatusOr<Feature> GetNodeFeature(const Node& node, const string& feature);

// Serialize the given program graph to an output stream.
[[nodiscard]] labm8::Status SerializeGraphVizToString(const ProgramGraph& graph,
                                                      std::ostream* ostream,
                                                      const NodeLabel& nodeLabelFormat = kText,
                                                      const string& nodeFeatureName = "");

}  // namespace format
}  // namespace graph
}  // namespace programl
