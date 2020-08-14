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

#include "absl/container/flat_hash_map.h"
#include "boost/graph/graphviz.hpp"
#include "labm8/cpp/logging.h"
#include "labm8/cpp/status.h"
#include "labm8/cpp/status_macros.h"
#include "labm8/cpp/string.h"
#include "programl/proto/program_graph.pb.h"

namespace error = labm8::error;

namespace programl {
namespace graph {
namespace format {

// The maximum length of a label. Labels longer than this are truncated with
// ellipses.
static const int kMaximumLabelLen = 32;

using AttributeMap = absl::flat_hash_map<string, string>;

using VertexProperties = boost::property<boost::vertex_attribute_t, AttributeMap>;
using EdgeProperties = boost::property<boost::edge_index_t, int,
                                       boost::property<boost::edge_attribute_t, AttributeMap>>;
using GraphProperties = boost::property<
    boost::graph_name_t, string,
    boost::property<boost::graph_graph_attribute_t, AttributeMap,
                    boost::property<boost::graph_vertex_attribute_t, AttributeMap,
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

namespace {

class GraphVizSerializer {
 public:
  GraphVizSerializer(const ProgramGraph& graph, const NodeLabel& nodeLabelFormat,
                     const string& nodeFeatureName)
      : graph_(graph), nodeLabelFormat_(nodeLabelFormat), nodeFeatureName_(nodeFeatureName) {}

 private:
  // Set global graphviz properties.
  void SetGraphvizProperties(boost::subgraph<GraphvizGraph>& main) const {
    // Set global graph, node, and edge style attributes.
    boost::get_property(main, boost::graph_graph_attribute)["margin"] = "0";
    boost::get_property(main, boost::graph_graph_attribute)["nodesep"] = "0.4";
    boost::get_property(main, boost::graph_graph_attribute)["ranksep"] = "0.4";
    boost::get_property(main, boost::graph_graph_attribute)["fontname"] = "Inconsolata";
    boost::get_property(main, boost::graph_graph_attribute)["fontsize"] = "20";

    boost::get_property(main, boost::graph_vertex_attribute)["fontname"] = "Inconsolata";
    boost::get_property(main, boost::graph_vertex_attribute)["fontsize"] = "20";
    boost::get_property(main, boost::graph_vertex_attribute)["penwidth"] = "2";
    boost::get_property(main, boost::graph_vertex_attribute)["width"] = "1";
    boost::get_property(main, boost::graph_vertex_attribute)["margin"] = "0";

    boost::get_property(main, boost::graph_edge_attribute)["fontname"] = "Inconsolata";
    boost::get_property(main, boost::graph_edge_attribute)["fontsize"] = "20";
    boost::get_property(main, boost::graph_edge_attribute)["penwidth"] = "3";
    boost::get_property(main, boost::graph_edge_attribute)["arrowsize"] = ".8";
  }

  std::vector<std::reference_wrapper<boost::subgraph<GraphvizGraph>>> MakeFunctionGraphs(
      boost::subgraph<GraphvizGraph>* main) {
    std::vector<std::reference_wrapper<boost::subgraph<GraphvizGraph>>> functionGraphs;
    for (int i = 0; i < graph_.function_size(); ++i) {
      const auto& function = graph_.function(i);

      functionGraphs.push_back(main->create_subgraph());
      auto& functionGraph = functionGraphs[functionGraphs.size() - 1].get();

      // Set the name of the function.
      string functionName = function.name();
      labm8::TruncateWithEllipsis(functionName, kMaximumLabelLen);
      boost::get_property(functionGraph, boost::graph_graph_attribute)["label"] = functionName;
      boost::get_property(functionGraph, boost::graph_graph_attribute)["margin"] = "10";
      boost::get_property(functionGraph, boost::graph_graph_attribute)["style"] = "dotted";

      // Set the name of the graph. Names must begin with "cluster".
      std::stringstream subgraphName;
      subgraphName << "cluster" << functionName;
      boost::get_property(functionGraphs[functionGraphs.size() - 1].get(), boost::graph_name) =
          subgraphName.str();
    }
    return functionGraphs;
  }

  string FeaturesToString(const Features& features) {
    std::stringstream os;
    const auto& it = features.feature().find(nodeFeatureName_);
    if (it == features.feature().end()) {
      // Do nothing if the feature is not found.
      return "";
    }
    const Feature& feature = it->second;

    // Int array
    for (int i = 0; i < std::min(feature.int64_list().value_size(), kMaximumLabelLen); ++i) {
      if (i) {
        os << ", ";
      }
      os << feature.int64_list().value(i);
    }
    // Float array
    os << std::setprecision(4);
    for (int i = 0; i < std::min(feature.float_list().value_size(), kMaximumLabelLen); ++i) {
      if (i) {
        os << ", ";
      }
      os << feature.float_list().value(i);
    }
    // Bytes array
    for (int i = 0; i < std::min(feature.bytes_list().value_size(), kMaximumLabelLen); ++i) {
      if (i) {
        os << ", ";
      }
      string value(feature.bytes_list().value(i));
      labm8::TruncateWithEllipsis(value, kMaximumLabelLen);
      os << value;
    }
    return os.str();
  }

  // Determine the text label for anode.
  string GetNodeLabel(const Node& node) {
    string text;
    switch (nodeLabelFormat_) {
      case kNone:
        break;
      case kText:
        text = node.text();
        break;
      case kFeature: {
        text = FeaturesToString(node.features());
        break;
      }
    }
    labm8::TruncateWithEllipsis(text, kMaximumLabelLen);
    return text;
  }

  template <typename T>
  void SetVertexAttributes(const Node& node, T& attributes) {
    attributes["label"] = GetNodeLabel(node);
    attributes["style"] = "filled";
    switch (node.type()) {
      case Node::INSTRUCTION:
        attributes["shape"] = "box";
        attributes["fillcolor"] = "#3c78d8";
        attributes["fontcolor"] = "#ffffff";
        break;
      case Node::VARIABLE:
        attributes["shape"] = "ellipse";
        attributes["fillcolor"] = "#f4cccc";
        attributes["color"] = "#990000";
        attributes["fontcolor"] = "#990000";
        break;
      case Node::CONSTANT:
        attributes["shape"] = "octagon";
        attributes["fillcolor"] = "#e99c9c";
        attributes["color"] = "#990000";
        attributes["fontcolor"] = "#990000";
        break;
      case Node::TYPE:
        attributes["shape"] = "diamond";
        attributes["fillcolor"] = "#cccccc";
        attributes["color"] = "#cccccc";
        attributes["fontcolor"] = "#222222";
        break;
      default:
        LOG(FATAL) << "unreachable";
    }
  }

  // Create the vertices.
  void CreateVertices(
      boost::subgraph<GraphvizGraph>* defaultGraph,
      std::vector<std::reference_wrapper<boost::subgraph<GraphvizGraph>>>* functionGraphs) {
    for (int i = 0; i < graph_.node_size(); ++i) {
      const Node& node = graph_.node(i);
      // Determine the subgraph to add this node to.
      boost::subgraph<GraphvizGraph>* dst = defaultGraph;
      if (i && (node.type() == Node::INSTRUCTION || node.type() == Node::VARIABLE)) {
        dst = &(*functionGraphs)[node.function()].get();
      }
      auto vertex = add_vertex(i, *dst);
      auto& attributes = get(boost::vertex_attribute, *dst)[vertex];
      SetVertexAttributes(node, attributes);
    }
  }

  template <typename T>
  void SetEdgeAttributes(const Edge& edge, T& attributes) {
    // Set the edge color.
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
      case Edge::TYPE:
        attributes["color"] = "#aaaaaa";
        attributes["weight"] = "1";
        attributes["penwidth"] = "1.5";
        break;
      default:
        LOG(FATAL) << "unreachable";
    }

    // Set the edge label.
    if (edge.position()) {
      // Position labels for control edge are drawn close to the originating
      // instruction. For control edges, they are drawn close to the branching
      // instruction. For data and type edges, they are drawn close to the
      // consuming node.
      const string label = edge.flow() == Edge::CONTROL ? "taillabel" : "headlabel";
      attributes[label] = std::to_string(edge.position());
      attributes["labelfontcolor"] = attributes["color"];
    }
  }

  // Add the edges to the graph.
  void CreateEdges(boost::subgraph<GraphvizGraph>* main) {
    for (int i = 0; i < graph_.edge_size(); ++i) {
      const Edge& edge = graph_.edge(i);
      auto newEdge = boost::add_edge(edge.source(), edge.target(), *main);
      auto& attributes = get(boost::edge_attribute, *main)[newEdge.first];
      SetEdgeAttributes(edge, attributes);
    }
  }

 public:
  labm8::Status Serialize(std::ostream* ostream) {
    // To construct a graphviz graph, we create a main graph and then produce
    // a subgraph for each function in the graph. Vertices (nodes) are then
    // added to the subgraphs, and edges added to the main graph.

    // Create a main graph and pre-allocate the number of nodes in the graph.
    // The main graph is used to store subgraphs which contain the actual
    // vertices. Vertices can't be added directly to the main graph.
    boost::subgraph<GraphvizGraph> main(graph_.node_size());
    boost::get_property(main, boost::graph_name) = "main";
    SetGraphvizProperties(main);

    // Since we can't add any vertices directly to the main graph, create a
    // subgraph for all nodes which do not have a function.
    boost::subgraph<GraphvizGraph>& external = main.create_subgraph();
    boost::get_property(external, boost::graph_name) = "external";

    auto functionGraphs = MakeFunctionGraphs(&main);
    CreateVertices(&external, &functionGraphs);
    CreateEdges(&main);

    boost::write_graphviz(*ostream, main);

    return labm8::Status::OK;
  }

 private:
  const ProgramGraph& graph_;
  const NodeLabel& nodeLabelFormat_;
  const string& nodeFeatureName_;
};

}  // anonymous namespace

labm8::Status SerializeGraphVizToString(const ProgramGraph& graph, std::ostream* ostream,
                                        const NodeLabel& nodeLabelFormat,
                                        const string& nodeFeatureName) {
  GraphVizSerializer serializer(graph, nodeLabelFormat, nodeFeatureName);
  return serializer.Serialize(ostream);
}

}  // namespace format
}  // namespace graph
}  // namespace programl
