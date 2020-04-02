// This file defines a class for constructing program graphs.
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

#include "deeplearning/ml4pl/graphs/graph_builder.h"

#include <utility>

#include "absl/container/flat_hash_set.h"
#include "labm8/cpp/logging.h"

namespace ml4pl {

GraphBuilder::GraphBuilder() : finalized_(false) {
  // Create the graph root node.
  auto node = AddNode(Node::STATEMENT);
  node.second->set_text("; root");
}

std::pair<int, Function*> GraphBuilder::AddFunction(const string& name) {
  CHECK(name.size()) << "Empty function name is invalid";
  int functionNumber = graph_.function_size();
  Function* function = graph_.add_function();
  function->set_name(name);
  return std::make_pair(functionNumber, function);
}

std::pair<int, Node*> GraphBuilder::AddStatement(const string& text,
                                                 int function) {
  auto node = AddNode(Node::STATEMENT);
  node.second->set_text(text);
  node.second->set_function(function);
  return node;
}

std::pair<int, Node*> GraphBuilder::AddIdentifier(const string& text,
                                                  int function) {
  auto node = AddNode(Node::IDENTIFIER);
  node.second->set_text(text);
  node.second->set_function(function);
  return node;
}

std::pair<int, Node*> GraphBuilder::AddImmediate(const string& text) {
  auto node = AddNode(Node::IMMEDIATE);
  node.second->set_text(text);
  return node;
}

void GraphBuilder::AddControlEdge(int sourceNode, int destinationNode) {
  CHECK(sourceNode < graph_.node_size())
      << "Source node " << sourceNode << " out of range for graph with "
      << graph_.node_size() << " nodes";
  CHECK(destinationNode < graph_.node_size())
      << "Destination node " << destinationNode
      << " out of range for graph with " << graph_.node_size() << " nodes";

  CHECK(graph_.node(sourceNode).type() == Node::STATEMENT &&
        graph_.node(destinationNode).type() == Node::STATEMENT)
      << "Control edge must connect statements";
  CHECK(graph_.node(sourceNode).function() ==
        graph_.node(destinationNode).function())
      << "Control edge must connect statements in the same function";

  control_adjacencies_[sourceNode].push_back(destinationNode);
}

void GraphBuilder::AddCallEdge(int sourceNode, int destinationNode) {
  CHECK(sourceNode < graph_.node_size())
      << "Source node " << sourceNode << " out of range for graph with "
      << graph_.node_size() << " nodes";
  CHECK(destinationNode < graph_.node_size())
      << "Destination node " << destinationNode
      << " out of range for graph with " << graph_.node_size() << " nodes";

  CHECK(graph_.node(sourceNode).type() == Node::STATEMENT &&
        graph_.node(destinationNode).type() == Node::STATEMENT)
      << "Call edge must connect statements";

  call_adjacencies_[sourceNode].insert(destinationNode);
}

void GraphBuilder::AddDataEdge(int sourceNode, int destinationNode,
                               int position) {
  CHECK(sourceNode < graph_.node_size())
      << "Source node " << sourceNode << " out of range for graph with "
      << graph_.node_size() << " nodes";
  CHECK(destinationNode < graph_.node_size())
      << "Destination node " << destinationNode
      << " out of range for graph with " << graph_.node_size() << " nodes";

  bool sourceIsData = (graph_.node(sourceNode).type() == Node::IDENTIFIER ||
                       graph_.node(sourceNode).type() == Node::IMMEDIATE);
  bool destinationIsData =
      (graph_.node(destinationNode).type() == Node::IDENTIFIER ||
       graph_.node(destinationNode).type() == Node::IMMEDIATE);

  CHECK(
      (sourceIsData &&
       graph_.node(destinationNode).type() == Node::STATEMENT) ||
      (graph_.node(sourceNode).type() == Node::STATEMENT && destinationIsData))
      << "Data edge must connect either a statement with data "
      << "OR data with a statement";

  data_reverse_adjacencies_[destinationNode].push_back({sourceNode, position});
}

void GraphBuilder::AddCallEdges(const size_t callingNode,
                                const FunctionEntryExits& calledFunction) {
  AddCallEdge(callingNode, calledFunction.first);
  for (auto exitNode : calledFunction.second) {
    AddCallEdge(exitNode, callingNode);
  }
}

void GraphBuilder::AddEdges(const std::vector<std::vector<size_t>>& adjacencies,
                            const Edge::Flow& flow,
                            std::vector<bool>* visitedNodes) {
  for (size_t sourceNode = 0; sourceNode < adjacencies.size(); ++sourceNode) {
    for (size_t position = 0; position < adjacencies[sourceNode].size();
         ++position) {
      size_t destinationNode = adjacencies[sourceNode][position];
      Edge* edge = graph_.add_edge();
      edge->set_flow(flow);
      edge->set_source_node(sourceNode);
      edge->set_destination_node(destinationNode);
      edge->set_position(position);

      // Record the source and destination nodes in the node set.
      (*visitedNodes)[sourceNode] = true;
      (*visitedNodes)[destinationNode] = true;
    }
  }
}

void GraphBuilder::AddEdges(
    const std::vector<absl::flat_hash_set<size_t>>& adjacencies,
    const Edge::Flow& flow, std::vector<bool>* visitedNodes) {
  for (size_t sourceNode = 0; sourceNode < adjacencies.size(); ++sourceNode) {
    for (size_t destinationNode : adjacencies[sourceNode]) {
      Edge* edge = graph_.add_edge();
      edge->set_flow(flow);
      edge->set_source_node(sourceNode);
      edge->set_destination_node(destinationNode);
      edge->set_position(0);

      // Record the source and destination nodes in the node set.
      (*visitedNodes)[sourceNode] = true;
      (*visitedNodes)[destinationNode] = true;
    }
  }
}

void GraphBuilder::AddReverseEdges(
    const std::vector<std::vector<std::pair<size_t, int>>>& adjacencies,
    const Edge::Flow& flow, std::vector<bool>* visitedNodes) {
  for (size_t destinationNode = 0; destinationNode < adjacencies.size();
       ++destinationNode) {
    // Track the positions to ensure that they are unique.
    absl::flat_hash_set<int> positionsSet;

    for (auto source : adjacencies[destinationNode]) {
      size_t sourceNode = source.first;
      int position = source.second;

      // Ensure that the position is unique.
      auto it = positionsSet.find(position);
      CHECK(it == positionsSet.end()) << "Duplicate position " << position;
      positionsSet.insert(position);

      Edge* edge = graph_.add_edge();
      edge->set_flow(flow);
      edge->set_source_node(sourceNode);
      edge->set_destination_node(destinationNode);
      edge->set_position(position);

      // Record the source and destination nodes in the node set.
      (*visitedNodes)[source.first] = true;
      (*visitedNodes)[destinationNode] = true;
    }
  }
}

const ProgramGraph& GraphBuilder::GetGraph() {
  if (finalized_) {
    return graph_;
  }
  std::vector<bool> visitedNodes(graph_.node_size(), false);

  AddEdges(control_adjacencies_, Edge::CONTROL, &visitedNodes);
  AddReverseEdges(data_reverse_adjacencies_, Edge::DATA, &visitedNodes);
  AddEdges(call_adjacencies_, Edge::CALL, &visitedNodes);

  // Check that all nodes except the root are connected. The root is allowed to
  // have no connections in the case where it is an empty graph.
  for (size_t i = 1; i < visitedNodes.size(); ++i) {
    CHECK(visitedNodes[i]) << "Graph contains node with no connections: "
                           << graph_.node(i).DebugString();
  }

  finalized_ = true;
  return graph_;
}

std::pair<int, Node*> GraphBuilder::AddNode(const Node::Type& type) {
  size_t nodeNumber = NextNodeNumber();
  Node* node = graph_.add_node();
  node->set_type(type);

  // Create empty adjacency lists for the new node.
  DCHECK(control_adjacencies_.size() == nodeNumber)
      << control_adjacencies_.size() << " != " << nodeNumber;
  DCHECK(data_reverse_adjacencies_.size() == nodeNumber)
      << data_reverse_adjacencies_.size() << " != " << nodeNumber;
  DCHECK(call_adjacencies_.size() == nodeNumber)
      << call_adjacencies_.size() << " != " << nodeNumber;

  control_adjacencies_.push_back({});
  data_reverse_adjacencies_.push_back({});
  call_adjacencies_.push_back({});

  return std::make_pair(nodeNumber, node);
}

}  // namespace ml4pl
