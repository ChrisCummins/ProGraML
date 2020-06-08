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

#include "programl/graph/format/graphviz_converter.h"

#include <iomanip>
#include <sstream>

#include "programl/proto/program_graph.pb.h"

#include "absl/container/flat_hash_map.h"
#include "boost/graph/graphviz.hpp"
#include "labm8/cpp/status.h"
#include "labm8/cpp/status_macros.h"
#include "labm8/cpp/string.h"

namespace error = labm8::error;

namespace programl {
namespace graph {
namespace format {

// The maximum length of a label. Labels longer than this are truncated with
// ellipses.
static const int kMaximumLabelLen = 32;

using AttributeMap = absl::flat_hash_map<string, string>;

using VertexProperties =
    boost::property<boost::vertex_attribute_t, AttributeMap>;
using EdgeProperties =
    boost::property<boost::edge_index_t, int,
                    boost::property<boost::edge_attribute_t, AttributeMap>>;
using GraphProperties = boost::property<
    boost::graph_name_t, string,
    boost::property<
        boost::graph_graph_attribute_t, AttributeMap,
        boost::property<
            boost::graph_vertex_attribute_t, AttributeMap,
            boost::property<boost::graph_edge_attribute_t, AttributeMap>>>>;

// An adjacency list for program graphs. We store the source Edge message as
// edge properties.
using GraphvizGraph = boost::adjacency_list<
    /*OutEdgeList=*/boost::vecS,
    /*VertexList=*/boost::vecS,
    /*Directed=*/boost::directedS,
    /*VertexProperties=*/VertexProperties,
    /*EdgeProperties=*/EdgeProperties,
    /*GraphProperties=*/GraphProperties>;

Status SerializeGraphVizToString(const ProgramGraph& graph,
                                 std::ostream* ostream,
                                 const NodeLabel& nodeLabelFormat,
                                 const string& nodeFeatureName) {
  // To construct a graphviz graph, we create a main graph and then produce
  // a subgraph for each function in the graph. Vertices (nodes) are then added
  // to the subgraphs, and edges added to the main graph.

  // Create a main graph and pre-allocate the number of nodes in the graph.
  // The main graph is used to store subgraphs which contain the actual
  // vertices. Vertices can't be added directly to the main graph.
  boost::subgraph<GraphvizGraph> main(graph.node_size());
  boost::get_property(main, boost::graph_name) = "main";

  // Set global graph, node, and edge style attributes.
  boost::get_property(main, boost::graph_graph_attribute)["margin"] = "0";
  boost::get_property(main, boost::graph_graph_attribute)["nodesep"] = "0.4";
  boost::get_property(main, boost::graph_graph_attribute)["ranksep"] = "0.4";
  boost::get_property(main, boost::graph_graph_attribute)["fontname"] =
      "Inconsolata";
  boost::get_property(main, boost::graph_graph_attribute)["fontsize"] = "20";

  boost::get_property(main, boost::graph_vertex_attribute)["fontname"] =
      "Inconsolata";
  boost::get_property(main, boost::graph_vertex_attribute)["fontsize"] = "20";
  boost::get_property(main, boost::graph_vertex_attribute)["penwidth"] = "2";
  boost::get_property(main, boost::graph_vertex_attribute)["width"] = "1";
  boost::get_property(main, boost::graph_vertex_attribute)["margin"] = "0";

  boost::get_property(main, boost::graph_edge_attribute)["fontname"] =
      "Inconsolata";
  boost::get_property(main, boost::graph_edge_attribute)["fontsize"] = "20";
  boost::get_property(main, boost::graph_edge_attribute)["penwidth"] = "3";
  boost::get_property(main, boost::graph_edge_attribute)["arrowsize"] = ".8";

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
    auto& functionGraph = functionGraphs[functionGraphs.size() - 1].get();

    // Set the name of the function.
    string functionName = function.name();
    labm8::TruncateWithEllipsis(functionName, kMaximumLabelLen);
    boost::get_property(functionGraph, boost::graph_graph_attribute)["label"] =
        functionName;
    boost::get_property(functionGraph, boost::graph_graph_attribute)["margin"] =
        "10";
    boost::get_property(functionGraph, boost::graph_graph_attribute)["style"] =
        "dotted";

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
    if (i && node.type() != Node::CONSTANT) {
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
      case kFeature: {
        std::stringstream os;
        const auto& it = node.features().feature().find(nodeFeatureName);
        if (it == node.features().feature().end()) {
          // Do nothing if the node is not found.
          break;
        }
        const Feature& feature = it->second;

        // Int array
        for (int i = 0; i < feature.int64_list().value_size(); ++i) {
          if (i) {
            os << ", ";
          }
          os << feature.int64_list().value(i);
        }
        // Float array
        os << std::setprecision(4);
        for (int i = 0; i < feature.float_list().value_size(); ++i) {
          if (i) {
            os << ", ";
          }
          os << feature.float_list().value(i);
        }
        // Bytes array
        for (int i = 0; i < feature.bytes_list().value_size(); ++i) {
          if (i) {
            os << ", ";
          }
          string value(feature.bytes_list().value(i));
          labm8::TruncateWithEllipsis(value, kMaximumLabelLen);
          os << value;
        }
        text = os.str();
        break;
      }
    }
    labm8::TruncateWithEllipsis(text, kMaximumLabelLen);
    attributes["label"] = text;

    // Set the node shape.
    switch (node.type()) {
      case Node::INSTRUCTION:
        attributes["shape"] = "box";
        attributes["style"] = "filled";
        attributes["fillcolor"] = "#3c78d8";
        attributes["fontcolor"] = "#ffffff";
        break;
      case Node::VARIABLE:
        attributes["shape"] = "ellipse";
        attributes["style"] = "filled";
        attributes["fillcolor"] = "#f4cccc";
        attributes["color"] = "#990000";
        attributes["fontcolor"] = "#990000";
        break;
      case Node::CONSTANT:
        attributes["shape"] = "diamond";
        attributes["style"] = "filled";
        attributes["fillcolor"] = "#e99c9c";
        attributes["color"] = "#990000";
        attributes["fontcolor"] = "#990000";
        break;
    }
  }

  // Add the edges to the graph.
  for (int i = 0; i < graph.edge_size(); ++i) {
    const Edge& edge = graph.edge(i);
    // Create the edge.
    auto newEdge = boost::add_edge(edge.source(), edge.target(), main);

    // Get the attributes dictionary for this edge.
    auto& attributes = get(boost::edge_attribute, main)[newEdge.first];

    // Set the edge color.
    string color;
    switch (edge.flow()) {
      case Edge::CONTROL:
        attributes["color"] = "#345393";
        attributes["weight"] = "10";
        break;
      case Edge::DATA:
        attributes["color"] = "#990000";
        attributes["weight"] = "0";
        break;
      case Edge::CALL:
        attributes["color"] = "#65ae4d";
        attributes["weight"] = "1";
        break;
    }

    // Set the edge label.
    if (edge.position()) {
      // Position labels for control edge are drawn close to the originating
      // instruction. For data edges, they are drawn closer to the consuming
      // instruction.
      const string label =
          edge.flow() == Edge::DATA ? "headlabel" : "taillabel";
      attributes[label] = std::to_string(edge.position());
      attributes["labelfontcolor"] = attributes["color"];
    }
  }

  boost::write_graphviz(*ostream, main);

  return Status::OK;
}

}  // namespace format
}  // namespace graph
}  // namespace programl
