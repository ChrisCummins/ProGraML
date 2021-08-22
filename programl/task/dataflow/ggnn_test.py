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
"""Run inference of a trained GGNN model on a single graph input.
"""
import pathlib
from typing import Any, Iterable, List
from os import listdir

import numpy as np
from labm8.py import app, pbutil
import networkx as nx
import matplotlib
matplotlib.use('Agg')  # to avoid using Xserver
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
from copy import deepcopy
#from sagemath.graphs.digraph import DiGraph, feedback_edge_set  # for removing cycles

from programl.graph.format.py.nx_format import ProgramGraphToNetworkX
from programl.models.base_graph_loader import BaseGraphLoader
from programl.models.batch_results import BatchResults
from programl.models.ggnn.ggnn import Ggnn
from programl.proto import (
    checkpoint_pb2,
    epoch_pb2,
    program_graph_features_pb2,
    program_graph_pb2,
)
from programl.task.dataflow import dataflow, vocabulary
from programl.task.dataflow.ggnn_batch_builder import DataflowGgnnBatchBuilder
from programl import serialize_ops

app.DEFINE_boolean(
    "cdfg",
    False,
    "If set, use the CDFG representation for programs. Defaults to ProGraML "
    "representations.",
)
app.DEFINE_boolean(
    "ig",
    False,
    "If set, run IG analysis.",
)
app.DEFINE_boolean(
    "batch",
    False,
    "If set, test samples in batch.",
)
app.DEFINE_boolean(
    "save_graph",
    False,
    "If set, save annotated graphs.",
)
app.DEFINE_boolean(
    "save_vis",
    False,
    "If set, save visualization images.",
)
app.DEFINE_boolean(
    "dep_guided_ig",
    False,
    "If set, enable dependency-guided IG attribution.",
)
app.DEFINE_boolean(
    "only_pred_y",
    False,
    "If set, only calculate IG attributions for pred_y=1 nodes.",
)
app.DEFINE_boolean(
    "use_acyclic_for_std_ig",
    False,
    "If set, use acyclicalized graphs to standard IG as well.",
)
app.DEFINE_integer(
    "max_vocab_size",
    0,
    "If > 0, limit the size of the vocabulary to this number.",
)
app.DEFINE_integer(
    "max_vis_graph_complexity",
    0,
    "If > 0, limit the max complexity of visualized graphs.",
)
app.DEFINE_integer(
    "max_removed_edges",
    -1,
    "If > -1, limit the max number of removed edges.",
)
app.DEFINE_float("target_vocab_cumfreq", 1.0, "The target cumulative frequency that.")
app.DEFINE_string(
    "ds_path",
    str(pathlib.Path("~/code-model-interpretation/ProGraML/dataset/dataflow").expanduser()),
    "The dataset directory.",
)
app.DEFINE_string("model", None, "The model checkpoint to restore")
app.DEFINE_string(
    "input",
    None,
    "Path of the input graph features list and index into it, "
    "e.g., /path/to/foo:1 to select the second graph from file /path/to/foo.",
)
FLAGS = app.FLAGS


class SingleGraphLoader(BaseGraphLoader):
    """`A graph loader which reads a single graph."""

    def __init__(
        self,
        graph: program_graph_pb2.ProgramGraph,
        features: program_graph_features_pb2.ProgramGraphFeatures,
    ):
        self.graph = graph
        self.features = features

    def IterableType(self) -> Any:
        return (
            program_graph_pb2.ProgramGraph,
            program_graph_features_pb2.ProgramGraphFeatures,
        )

    def __iter__(self) -> Iterable["IterableType"]:
        yield self.graph, self.features

    def Stop(self):
        pass


class TooComplexGraphError(Exception):
    pass


class TooManyRootNodesError(Exception):
    pass


class CycleInGraphError(Exception):
    pass


class TooManyEdgesRemoved(Exception):
    pass


def RemoveCyclesFromGraphSimple(
    graph: nx.DiGraph,
) -> nx.DiGraph:
    # This is an overly simplied solution, without either correctness or optimality
    # guarantee. The original minimum feeback arc set problem is NP-hard and has better
    # approximation algorithms. I will replace this function in the future.
    while not nx.algorithms.dag.is_directed_acyclic_graph(graph):
        cycle = nx.find_cycle(graph)
        for edge in cycle:
            tail, head, key = edge
            edge_data = graph.get_edge_data(tail, head, key=key)
            graph.remove_edge(tail, head, key=key)
            print("Removed an edge: (%d --> %d)" % (tail, head))
            if not nx.algorithms.components.is_weakly_connected(graph):
                graph.add_edge(tail, head, key=key)
                for attr_key, attr_value in edge_data.items():
                    graph[tail][head][key][attr_key] = attr_value
                print("Added back an edge: (%d --> %d)" % (tail, head))
    return graph


def CalculateInterpolationOrderFromGraph(
    graph: program_graph_pb2.ProgramGraph,
    use_simple_removal: bool = True,
    reverse: bool = False,
    max_removed_edges int = -1,
) -> List[int]:
    # This function returns the (topological) order of nodes to evaluate
    # for interpolations in IG
    networkx_graph = ProgramGraphToNetworkX(graph)
    is_acyclic = nx.algorithms.dag.is_directed_acyclic_graph(networkx_graph)
    if is_acyclic:
        ordered_nodes = nx.topological_sort(networkx_graph)
        return list(ordered_nodes)
    else:
        # Cycle(s) detected and we need to remove them now
        if not use_simple_removal:
            sage_graph = DiGragh(networkx_graph)
            acyclic_sage_graph = sage_graph.feedback_edge_set()
            acyclic_networkx_graph = acyclic_sage_graph.networkx_graph()
        else:
            acyclic_networkx_graph = RemoveCyclesFromGraphSimple(networkx_graph)
            if max_removed_edges != -1:
                original_num_edges = len(networkx_graph.edges)
                trimmed_num_edges = len(acyclic_networkx_graph.edges)
                num_removed_edges = original_num_edges - trimmed_num_edges
                if num_removed_edges > max_removed_edges:
                    raise TooManyEdgesRemoved

        # Sanity check, only return the graph if it is acyclic
        is_acyclic = nx.algorithms.dag.is_directed_acyclic_graph(acyclic_networkx_graph)
        if not is_acyclic:
            raise CycleInGraphError
        else:
            ordered_nodes = nx.topological_sort(acyclic_networkx_graph)
            return list(ordered_nodes)


def TestOne(
    features_list_path: pathlib.Path,
    features_list_index: int,
    checkpoint_path: pathlib.Path,
    ds_path: str,
    run_ig: bool,
    max_vis_graph_complexity: int,
    dep_guided_ig: bool,
    all_nodes_out: bool,
    max_removed_edges: int,
) -> BatchResults:
    if dep_guided_ig and not run_ig:
        print("run_ig and dep_guided_ig args take different values which is invalid!")
        raise RuntimeError

    path = pathlib.Path(ds_path)
    
    features_list = pbutil.FromFile(
        features_list_path,
        program_graph_features_pb2.ProgramGraphFeaturesList(),
    )
    features = features_list.graph[features_list_index]

    graph_name = features_list_path.name[: -len(".ProgramGraphFeaturesList.pb")]
    graph = pbutil.FromFile(
        path / "graphs" / f"{graph_name}.ProgramGraph.pb",
        program_graph_pb2.ProgramGraph(),
    )

    # First, we need to fix empty node features
    graph = FixEmptyNodeFeatures(graph, features)
    
    if dep_guided_ig:
        interpolation_order = CalculateInterpolationOrderFromGraph(graph, max_removed_edges=max_removed_edges)
    else:
        interpolation_order = None

    if max_vis_graph_complexity != 0:
        if (len(graph.node)) > max_vis_graph_complexity:
            raise TooComplexGraphError
        if (len(graph.edge)) > max_vis_graph_complexity:
            raise TooComplexGraphError

    num_root_nodes = 0
    if all_nodes_out:
        nodes_out = []
        for i in range(len(graph.node)):
            if features.node_features.feature_list["data_flow_root_node"].feature[i].int64_list.value == [1]:
                num_root_nodes +=1 
            if features.node_features.feature_list["data_flow_value"].feature[i].int64_list.value == [1]:
                nodes_out.append(i)
    else:
        for i in range(len(graph.node)):
            if features.node_features.feature_list["data_flow_root_node"].feature[i].int64_list.value == [1]:
                num_root_nodes +=1
    if num_root_nodes > 1:
        raise TooManyRootNodesError

    # Instantiate and restore the model.
    vocab = vocabulary.LoadVocabulary(
        path,
        model_name="cdfg" if FLAGS.cdfg else "programl",
        max_items=FLAGS.max_vocab_size,
        target_cumfreq=FLAGS.target_vocab_cumfreq,
    )

    if FLAGS.cdfg:
        FLAGS.use_position_embeddings = False

    model = Ggnn(
        vocabulary=vocab,
        test_only=True,
        node_y_dimensionality=2,
        graph_y_dimensionality=0,
        graph_x_dimensionality=0,
        use_selector_embeddings=True,
    )
    checkpoint = pbutil.FromFile(checkpoint_path, checkpoint_pb2.Checkpoint())
    model.RestoreCheckpoint(checkpoint)

    batch = list(
        DataflowGgnnBatchBuilder(
            graph_loader=SingleGraphLoader(graph=graph, features=features),
            vocabulary=vocab,
            max_node_size=int(1e9),
            use_cdfg=FLAGS.cdfg,
            max_batch_count=1,
        )
    )[0]
    
    if all_nodes_out:
        results_predicted_nodes = []
        for node_out in nodes_out:
            results = model.RunBatch(
                epoch_pb2.TEST, 
                batch, 
                run_ig=run_ig, 
                dep_guided_ig=dep_guided_ig, 
                interpolation_order=interpolation_order,
                node_out=node_out,
            )
            results_predicted_nodes.append(results)
        return AnnotateGraphWithBatchResultsForPredictedNodes(graph, features, results_predicted_nodes, nodes_out, run_ig)
    else:
        results = model.RunBatch(
            epoch_pb2.TEST, 
            batch, 
            run_ig=run_ig, 
            dep_guided_ig=dep_guided_ig, 
            interpolation_order=interpolation_order,
        )
        return AnnotateGraphWithBatchResults(graph, features, results, run_ig)



def FixEmptyNodeFeatures(
    graph: program_graph_pb2.ProgramGraph,
    features: program_graph_features_pb2.ProgramGraphFeatures,
) -> program_graph_pb2.ProgramGraph:
    assert len(graph.node) == len(
        features.node_features.feature_list["data_flow_value"].feature
    )
    assert len(graph.node) == len(
        features.node_features.feature_list["data_flow_root_node"].feature
    )

    for i, node in enumerate(graph.node):
        # Fix empty node feature errors so that we can persist graphs
        if node.features.feature["full_text"].bytes_list.value == []:
            node.features.feature["full_text"].bytes_list.value.append(b'')

    return graph


def AnnotateGraphWithBatchResultsForPredictedNodes(
    base_graph: program_graph_pb2.ProgramGraph,
    features: program_graph_features_pb2.ProgramGraphFeatures,
    results_predicted_nodes: List[BatchResults],
    nodes_out: List[int],
    run_ig: bool,
) -> program_graph_pb2.ProgramGraph:
    """Annotate graph with features describing the target labels and predicted outcomes."""
    assert len(base_graph.node) == len(
        features.node_features.feature_list["data_flow_value"].feature
    )
    assert len(base_graph.node) == len(
        features.node_features.feature_list["data_flow_root_node"].feature
    )

    graphs = []
    
    for i in range(len(results_predicted_nodes)):
        results = results_predicted_nodes[i]
        graph = deepcopy(base_graph)
        if run_ig:
            assert len(graph.node) == results.attributions.shape[0]

        true_y = np.argmax(results.targets, axis=1)
        pred_y = np.argmax(results.predictions, axis=1)

        for j, node in enumerate(graph.node):
            node.features.feature["data_flow_root_node"].CopyFrom(
                features.node_features.feature_list["data_flow_root_node"].feature[j]
            )
            if j == nodes_out[i]:
                node.features.feature["true_y"].int64_list.value.append(true_y[0])
                node.features.feature["pred_y"].int64_list.value.append(pred_y[0])
            else:
                node.features.feature["true_y"].int64_list.value.append(0)
                node.features.feature["pred_y"].int64_list.value.append(0)
            if run_ig:
                node.features.feature["attribution_order"].int64_list.value.append(results.attributions[j])

        graph.features.feature["loss"].float_list.value.append(results.loss)
        graph.features.feature["accuracy"].float_list.value.append(results.accuracy)
        graph.features.feature["precision"].float_list.value.append(results.precision)
        graph.features.feature["recall"].float_list.value.append(results.recall)
        graph.features.feature["f1"].float_list.value.append(results.f1)
        graph.features.feature["confusion_matrix"].int64_list.value[:] = np.hstack(
            results.confusion_matrix
        )

        graphs.append(graph)

    return graphs


def AnnotateGraphWithBatchResults(
    graph: program_graph_pb2.ProgramGraph,
    features: program_graph_features_pb2.ProgramGraphFeatures,
    results: BatchResults,
    run_ig: bool,
) -> program_graph_pb2.ProgramGraph:
    """Annotate graph with features describing the target labels and predicted outcomes."""
    assert len(graph.node) == len(
        features.node_features.feature_list["data_flow_value"].feature
    )
    assert len(graph.node) == len(
        features.node_features.feature_list["data_flow_root_node"].feature
    )
    assert len(graph.node) == results.targets.shape[0]
    if run_ig:
        assert len(graph.node) == results.attributions.shape[0]

    true_y = np.argmax(results.targets, axis=1)
    pred_y = np.argmax(results.predictions, axis=1)

    for i, node in enumerate(graph.node):
        node.features.feature["data_flow_value"].CopyFrom(
            features.node_features.feature_list["data_flow_value"].feature[i]
        )
        node.features.feature["data_flow_root_node"].CopyFrom(
            features.node_features.feature_list["data_flow_root_node"].feature[i]
        )
        node.features.feature["target"].float_list.value[:] = results.targets[i]
        node.features.feature["prediction"].float_list.value[:] = results.predictions[i]
        node.features.feature["true_y"].int64_list.value.append(true_y[i])
        node.features.feature["pred_y"].int64_list.value.append(pred_y[i])
        node.features.feature["correct"].int64_list.value.append(true_y[i] == pred_y[i])
        if run_ig:
            node.features.feature["attribution_order"].int64_list.value.append(results.attributions[i])

    graph.features.feature["loss"].float_list.value.append(results.loss)
    graph.features.feature["accuracy"].float_list.value.append(results.accuracy)
    graph.features.feature["precision"].float_list.value.append(results.precision)
    graph.features.feature["recall"].float_list.value.append(results.recall)
    graph.features.feature["f1"].float_list.value.append(results.f1)
    graph.features.feature["confusion_matrix"].int64_list.value[:] = np.hstack(
        results.confusion_matrix
    )

    return graph


def TestOneGraph(
    ds_path, 
    model_path, 
    graph_path, 
    graph_idx, 
    max_vis_graph_complexity,
    run_ig=False,
    dep_guided_ig=False,
    all_nodes_out=False,
    max_removed_edges=-1,
):
    if all_nodes_out:
        graphs = TestOne(
            features_list_path=pathlib.Path(graph_path),
            features_list_index=int(graph_idx),
            checkpoint_path=pathlib.Path(ds_path + model_path),
            ds_path=ds_path,
            run_ig=run_ig,
            max_vis_graph_complexity=max_vis_graph_complexity,
            dep_guided_ig=dep_guided_ig,
            all_nodes_out=all_nodes_out,
            max_removed_edges=max_removed_edges,
        )
        return graphs
    else:
        graph = TestOne(
            features_list_path=pathlib.Path(graph_path),
            features_list_index=int(graph_idx),
            checkpoint_path=pathlib.Path(ds_path + model_path),
            ds_path=ds_path,
            run_ig=run_ig,
            max_vis_graph_complexity=max_vis_graph_complexity,
            dep_guided_ig=dep_guided_ig,
            all_nodes_out=all_nodes_out,
            max_removed_edges=max_removed_edges,
        )
        return graph


def DrawAndSaveGraph(
    graph, 
    ds_path, 
    graph_fname, 
    save_graph=False,
    save_vis=False,
    suffix=''
):
    if not isinstance(graph, list):
        # Meaning we are handling per-node IG
        graphs = [graph]
    else:
        graphs = graph

    for i in range(len(graphs)):
        graph = graphs[i]
        save_graph_path = ds_path + '/vis_res/' + graph_fname + ".AttributedProgramGraphFeaturesList.%s.%d.pb" % (suffix, i)
        if save_graph:
            print("Saving annotated graph to %s..." % save_graph_path)
            serialize_ops.save_graphs(save_path, [graph])
        
        networkx_graph = ProgramGraphToNetworkX(graph)
        
        original_labels = nx.get_node_attributes(networkx_graph, "features")

        labels = {}
        for node, features in original_labels.items():
            curr_label = ""
            curr_label += "Pred: " + str(features["pred_y"]) + " | "
            curr_label += "True: " + str(features["true_y"]) + " | \n"
            curr_label += "Attr: " + str(features["attribution_order"]) + " | "
            if features["data_flow_root_node"] == 0:
                curr_label += "Target"
            labels[node] = '[' + curr_label + ']'

        color = []
        for node in networkx_graph.nodes():
            if original_labels[node]["data_flow_root_node"] == [1]:
                color.append('red')
            elif original_labels[node]["pred_y"] == [1]:
                color.append('purple')
            else:
                color.append('grey')

        pos = graphviz_layout(networkx_graph, prog='neato')
        nx.draw(networkx_graph, pos=pos, labels=labels, node_size=500, node_color=color)
        
        if save_vis:
            save_img_path = ds_path + '/vis_res/' + graph_fname + ".AttributedProgramGraph.%s.%d.png" % (suffix, i)
            print("Saving visualization of annotated graph to %s..." % save_img_path)
            plt.show()
            plt.savefig(save_img_path, format="PNG")
            plt.clf()


def Main():
    """Main entry point."""
    dataflow.PatchWarnings()

    if FLAGS.batch:
        graphs_dir = FLAGS.ds_path + '/labels/datadep/'
        graph_fnames = listdir(graphs_dir)
        for graph_fname in graph_fnames:
            try:
                original_graph_fname = graph_fname[: -len(".ProgramGraphFeaturesList.pb")].split('/')[-1]
                print("Processing graph file: %s..." % graph_fname)
                graph_path = graphs_dir + graph_fname
                try:
                    graph_std_ig = TestOneGraph(
                        FLAGS.ds_path, 
                        FLAGS.model, 
                        graph_path, 
                        '-1', 
                        FLAGS.max_vis_graph_complexity,
                        run_ig=FLAGS.ig,
                        dep_guided_ig=False,
                        all_nodes_out=FLAGS.only_pred_y,
                        max_removed_edges=FLAGS.max_removed_edges,
                    )
                    graph_dep_guided_ig = TestOneGraph(
                        FLAGS.ds_path,
                        FLAGS.model,
                        graph_path, 
                        '-1', 
                        FLAGS.max_vis_graph_complexity,
                        run_ig=FLAGS.ig,
                        dep_guided_ig=True,
                        all_nodes_out=FLAGS.only_pred_y,
                        max_removed_edges=FLAGS.max_removed_edges,
                    )
                    print("Acyclic graph found and loaded.")
                except TooComplexGraphError:
                    print("Skipping graph %s due to exceeding number of nodes..." % original_graph_fname)
                    continue
                except CycleInGraphError:
                    print("Skipping graph %s due to presence of graph cycle(s)..." % original_graph_fname)
                    continue
                except TooManyEdgesRemoved:
                    print("Skipping graph %s due to exceeding number of removed edges..." % original_graph_fname)
                    continue

                if FLAGS.ig and FLAGS.dep_guided_ig:
                    DrawAndSaveGraph(
                        graph_std_ig, FLAGS.ds_path,
                        original_graph_fname, save_graph=FLAGS.save_graph, 
                        save_vis=FLAGS.save_vis, suffix='std_ig'
                    )
                    DrawAndSaveGraph(
                        graph_dep_guided_ig, FLAGS.ds_path,
                        original_graph_fname, save_graph=FLAGS.save_graph, 
                        save_vis=FLAGS.save_vis, suffix='dep_guided_ig'
                    )
            except Exception as err:
                print("Error testing %s -- %s" % (graph_fname, str(err)))
                continue
    else:
        features_list_path, features_list_index = FLAGS.input.split(":")
        graph_fname = features_list_path[: -len(".ProgramGraphFeaturesList.pb")].split('/')[-1]
        try:
            graph_std_ig = TestOneGraph(
                FLAGS.ds_path, 
                FLAGS.model, 
                FLAGS.ds_path + features_list_path, 
                features_list_index, 
                FLAGS.max_vis_graph_complexity,
                run_ig=FLAGS.ig,
                dep_guided_ig=False,
                all_nodes_out=FLAGS.only_pred_y,
            )
            graph_dep_guided_ig = TestOneGraph(
                FLAGS.ds_path,
                FLAGS.model,
                FLAGS.ds_path + features_list_path, 
                features_list_index, 
                FLAGS.max_vis_graph_complexity,
                run_ig=FLAGS.ig,
                dep_guided_ig=True,
                all_nodes_out=FLAGS.only_pred_y,
            )
        except TooComplexGraphError:
            print("Skipping graph %s due to exceeding number of nodes..." % graph_fname)
            exit()
        except CycleInGraphError:
            print("Skipping graph %s due to presence of graph cycle(s)..." % graph_fname)
            exit()

        if FLAGS.ig and FLAGS.dep_guided_ig:
            DrawAndSaveGraph(
                graph_std_ig, FLAGS.ds_path, 
                graph_fname, save_graph=FLAGS.save_graph, 
                save_vis=FLAGS.save_vis, suffix='std_ig'
            )
            DrawAndSaveGraph(
                graph_dep_guided_ig, FLAGS.ds_path, 
                graph_fname, save_graph=FLAGS.save_graph, 
                save_vis=FLAGS.save_vis, suffix='dep_guided_ig'
            )


if __name__ == "__main__":
    app.Run(Main)
