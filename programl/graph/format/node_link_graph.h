// This file defines functions for converting program graphs to node link JSON.
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
#include "nlohmann/json.hpp"
#include "programl/proto/program_graph.pb.h"

namespace programl {
namespace graph {
namespace format {

// Convert a ProgramGraph protocol message to a JSON dictionary containing a
// node link graph, suitable for loading with networkx. Conversion to node link
// graph is lossless.
//
// See:
// https://networkx.github.io/documentation/stable/reference/readwrite/json_graph.html
[[nodiscard]] labm8::Status ProgramGraphToNodeLinkGraph(const ProgramGraph& graph,
                                                        nlohmann::json* dict);

namespace detail {

// Helpers for converting from program graph to JSON node link graph.
[[nodiscard]] labm8::Status CreateGraphDict(const ProgramGraph& graph, nlohmann::json* graphDict);
[[nodiscard]] labm8::Status CreateNodesList(const ProgramGraph& graph, nlohmann::json* nodes);
[[nodiscard]] labm8::Status CreateLinksList(const ProgramGraph& graph, nlohmann::json* links);
[[nodiscard]] labm8::Status CreateFeaturesDict(const Features& features,
                                               nlohmann::json* featuresDict);
[[nodiscard]] labm8::Status CreateFeatureArray(const Feature& feature,
                                               nlohmann::json* featureArray);

}  // namespace detail

}  // namespace format
}  // namespace graph
}  // namespace programl
