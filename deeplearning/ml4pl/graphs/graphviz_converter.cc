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

#include "deeplearning/ml4pl/graphs/graphviz_converter.h"

#include <iomanip>
#include <sstream>

#include "absl/container/flat_hash_map.h"
#include "boost/graph/graphviz.hpp"
#include "labm8/cpp/logging.h"
#include "labm8/cpp/string.h"

namespace ml4pl {

namespace {

// An enum describing the different node attributes that can be used for setting
// text labels.
enum NodeLabel { kNone = 0, kText, kPreprocessedText, kX, kY };

// Convert a string node labels value to a NodeLabel enum.
NodeLabel NodeLabelFromString(const string& value) {
  if (value == "none") {
    return NodeLabel::kNone;
  } else if (value == "text") {
    return NodeLabel::kText;
  } else if (value == "preprocessed_text") {
    return NodeLabel::kPreprocessedText;
  } else if (value == "x") {
    return NodeLabel::kX;
  } else if (value == "y") {
    return NodeLabel::kY;
  }

  LOG(FATAL) << "Unknown value for node label: `" << value << "`. Supported "
             << "values: none,text,preprocessed_text,x,y";
}

}  // anonymous namespace

// The maximum length of a label. Labels longer than this are truncated with
// ellipses.
static const int kMaximumLabelLen = 32;

using AttributeMap = absl::flat_hash_map<string, string>;

// An adjacency list for program graphs. We store the source Edge message as
// edge properties.
using GraphvizGraph = boost::adjacency_list<
    /*OutEdgeList=*/boost::vecS,
    /*VertexList=*/boost::vecS,
    /*Directed=*/boost::directedS,
    /*VertexProperties=*/
    boost::property<boost::vertex_attribute_t, AttributeMap>,
    /*EdgeProperties=*/
    boost::property<boost::edge_index_t, int,
                    boost::property<boost::edge_attribute_t, AttributeMap>>,
    /*GraphProperties=*/
    boost::property<
        boost::graph_name_t, string,
        boost::property<
            boost::graph_graph_attribute_t, AttributeMap,
            boost::property<boost::graph_vertex_attribute_t, AttributeMap,
                            boost::property<boost::graph_edge_attribute_t,
                                            AttributeMap>>>>>;

void SerializeGraphVizToString(const ProgramGraph& graph, std::ostream* ostream,
                               const string& nodeLabels) {
  NodeLabel nodeLabelFormat = NodeLabelFromString(nodeLabels);

  // To construct a graphviz graph, we create a main graph and then produce
  // a subgraph for each function in the graph. Vertices (nodes) are then added
  // to the subgraphs, and edges added to the main graph.

  // Create a main graph and pre-allocate the number of nodes in the graph.
  // The main graph is used to store subgraphs which contain the actual
  // vertices. Vertices can't be added directly to the main graph.
  boost::subgraph<GraphvizGraph> main(graph.node_size());
  boost::get_property(main, boost::graph_name) = "main";

  // Since we can't add any vertices directly to the main graph, create a
  // subgraph for all nodes which do not have a function.
  boost::subgraph<GraphvizGraph>& external = main.create_subgraph();
  boost::get_property(external, boost::graph_name) = "external";

  // Generate a list of per-function subgraphs.
  std::vector<std::reference_wrapper<boost::subgraph<GraphvizGraph>>>
      functionGraphs;
  for (int i = 0; i < graph.function_size(); ++i) {
    const auto& function = graph.function(i);

    functionGraphs.push_back(main.create_subgraph());

    // Set the name of the function.
    string functionName = function.name();
    labm8::TruncateWithEllipsis(functionName, kMaximumLabelLen);
    boost::get_property(functionGraphs[functionGraphs.size() - 1].get(),
                        boost::graph_graph_attribute)["label"] = functionName;

    // Set the name of the graph. Names must begin with "cluster".
    std::stringstream subgraphName;
    subgraphName << "cluster" << functionName;
    boost::get_property(functionGraphs[functionGraphs.size() - 1].get(),
                        boost::graph_name) = subgraphName.str();
  }

  // Create the vertices.
  for (int i = 0; i < graph.node_size(); ++i) {
    const Node& node = graph.node(i);

    // Determine the subgraph to add this node to.
    boost::subgraph<GraphvizGraph>* dst = &external;
    if (node.has_function()) {
      dst = &functionGraphs[node.function()].get();
    }

    // Create the vertex.
    auto vertex = add_vertex(i, *dst);

    // Get the attributes dictionary for this vertex.
    auto& attributes = get(boost::vertex_attribute, *dst)[vertex];

    // Set the node text.
    std::stringstream textStream;
    string text;
    switch (nodeLabelFormat) {
      case kNone:
        break;
      case kText:
        text = node.text();
        break;
      case kPreprocessedText:
        LOG(INFO) << "TEXT " << node.text() << " " << node.DebugString();
        text = node.preprocessed_text();
        break;
      case kX:
        textStream << std::setprecision(3);
        for (int i = 0; i < node.x_size(); ++i) {
          if (i) {
            textStream << " ";
          }
          textStream << node.x(i);
        }
        text = textStream.str();
        break;
      case kY:
        textStream << std::setprecision(3);
        for (int i = 0; i < node.y_size(); ++i) {
          if (i) {
            textStream << " ";
          }
          textStream << node.y(i);
        }
        text = textStream.str();
        break;
    }
    labm8::TruncateWithEllipsis(text, kMaximumLabelLen);
    attributes["label"] = text;

    // Set the node shape.
    string shape;
    switch (node.type()) {
      case Node::STATEMENT:
        shape = "box";
        break;
      case Node::IDENTIFIER:
        shape = "ellipse";
        break;
      case Node::IMMEDIATE:
        shape = "ellipse";
        break;
    }

    // The root node has a special shape.
    if (!i) {
      shape = "doubleoctagon";
    }
    attributes["shape"] = shape;
  }

  // Add the edges to the graph.
  for (int i = 0; i < graph.edge_size(); ++i) {
    const Edge& edge = graph.edge(i);
    // Create the edge.
    auto newEdge =
        boost::add_edge(edge.source_node(), edge.destination_node(), main);

    // Get the attributes dictionary for this edge.
    auto& attributes = get(boost::edge_attribute, main)[newEdge.first];

    // Set the edge color.
    string color;
    switch (edge.flow()) {
      case Edge::CONTROL:
        color = "blue";
        break;
      case Edge::DATA:
        color = "red";
        break;
      case Edge::CALL:
        color = "green";
        break;
    }
    attributes["color"] = color;

    // Set the edge label.
    if (edge.position()) {
      attributes["label"] = std::to_string(edge.position());
    }
  }

  boost::write_graphviz(*ostream, main);
}

}  // namespace ml4pl
