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

#include "programl/graph/analysis/subexpressions.h"

#include <sstream>
#include <utility>
#include <vector>

#include "labm8/cpp/logging.h"
#include "labm8/cpp/status.h"
#include "programl/graph/features.h"

using labm8::Status;
using std::vector;
namespace error = labm8::error;

namespace programl {
namespace graph {
namespace analysis {

Status SubexpressionsAnalysis::Init() {
  ComputeAdjacencies({.control = false,
                      .reverse_control = false,
                      .data = false,
                      .reverse_data = true,
                      .reverse_data_positions = true});

  const auto& rdfg = adjacencies().reverse_data;
  DCHECK(rdfg.size() == graph().node_size())
      << "Reverse data-flow graph size: " << rdfg.size() << " != "
      << " graph size: " << graph().node_size();
  const auto& rdp = adjacencies().reverse_data_positions;
  DCHECK(rdp.size() == graph().node_size())
      << "Reverse data-flow edge positions size: " << rdp.size() << " != "
      << " graph size: " << graph().node_size();

  // An expression is a statement and an ordered list of operand nodes.
  using Expression = std::pair<string, vector<int>>;

  // A map from expression to a list of nodes which evaluate this expression.
  absl::flat_hash_map<Expression, absl::flat_hash_set<int>> subexpressionSets;

  // Start at index 1 to skip the graph root.
  for (int i = 1; i < graph().node_size(); ++i) {
    const auto& node = graph().node(i);

    // We only care about instructions.
    if (node.type() != Node::INSTRUCTION) {
      continue;
    }

    // Build a list of operand <position, operand> pairs.
    using Operand = std::pair<int, int>;
    vector<Operand> positionOrderedPairs;
    positionOrderedPairs.reserve(rdfg[i].size());
    for (int j = 0; j < rdfg[i].size(); ++j) {
      positionOrderedPairs.emplace_back(rdp[i][j], rdfg[i][j]);
    }

    // A statement without operands can never be a common subexpression.
    if (positionOrderedPairs.empty()) {
      continue;
    }

    // If the operands are commutative, sort operands by their name. For,
    // non-commutative operands, sort the operand position (i.e. order).
    // E.g.
    //    '%3 = add %2 %1' == '%4 = add %1 %2'  # commutative
    // but:
    //    '%3 = sub %2 %1' != '%4 = sub %1 %2'  # non-commutative
    //
    // Commutative statements derived from:
    // <https://llvm.org/docs/LangRef.html#instruction-reference>.
    const auto& opcodeName = node.text();
    if (
        // Binary operators:
        opcodeName == "add" || opcodeName == "fadd" || opcodeName == "mul" ||
        opcodeName == "fmul" ||
        // Bitwise binary operators:
        opcodeName == "or" || opcodeName == "and" || opcodeName == "xor") {
      // Commutative statement, order by identifier name.
      std::sort(positionOrderedPairs.begin(), positionOrderedPairs.end(),
                [](const Operand& a, const Operand& b) { return a.second < b.second; });
    } else {
      // Non-commutative statement, order by position.
      std::sort(positionOrderedPairs.begin(), positionOrderedPairs.end(),
                [](const Operand& a, const Operand& b) { return a.first < b.first; });
    }

    // Now that we have ordered the list, strip the order key to make a list
    // of node indices.
    vector<int> operands(positionOrderedPairs.size());
    for (size_t j = 0; j < positionOrderedPairs.size(); ++j) {
      operands[j] = positionOrderedPairs[j].second;
    }

    // An expression is an instruction and an ordered list of operand node
    // indices.
    Expression expression{opcodeName, operands};

    // Add the current node to the expression set table.
    subexpressionSets[expression].insert(i);
  }

  for (const auto& it : subexpressionSets) {
    if (it.second.size() > 1) {
      subexpressionSets_.push_back(it.second);
    }
  }

  return Status::OK;
}

vector<int> SubexpressionsAnalysis::GetEligibleRootNodes() {
  vector<int> rootNodes;
  for (const auto& set : subexpression_sets()) {
    for (const auto& v : set) {
      rootNodes.push_back(v);
    }
  }
  return rootNodes;
}

Status SubexpressionsAnalysis::RunOne(int rootNode, ProgramGraphFeatures* features) {
  for (const auto& expressionSet : subexpressionSets_) {
    if (expressionSet.contains(rootNode)) {
      Feature falseFeature = CreateFeature(0);
      Feature trueFeature = CreateFeature(1);

      for (int i = 0; i < graph().node_size(); ++i) {
        AddNodeFeature(features, "data_flow_root_node", i == rootNode ? trueFeature : falseFeature);
        AddNodeFeature(features, "data_flow_value",
                       expressionSet.contains(i) ? trueFeature : falseFeature);
      }

      SetFeature(features->mutable_features(), "data_flow_step_count", CreateFeature(2));
      SetFeature(features->mutable_features(), "data_flow_active_node_count",
                 CreateFeature(expressionSet.size()));

      return Status::OK;
    }
  }

  return Status::OK;
}

string SubexpressionsAnalysis::ToString() const {
  std::stringstream o;
  o << *this;
  return o.str();
}

std::ostream& operator<<(std::ostream& os, const SubexpressionsAnalysis& s) {
  os << "Adjacencies:" << std::endl
     << s.adjacencies() << "#. expression sets: " << s.subexpression_sets().size() << std::endl;
  for (int i = 0; i < s.subexpression_sets().size(); ++i) {
    os << "Expression set " << i << ": {";
    int j = 0;
    for (const auto& v : s.subexpression_sets()[i]) {
      if (j) {
        os << ", ";
      }
      os << v;
      ++j;
    }
    os << "}" << std::endl;
  }
  return os;
}

}  // namespace analysis
}  // namespace graph
}  // namespace programl
