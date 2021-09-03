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
import numpy as np
from labm8.py import app, pbutil
import networkx as nx
import matplotlib
matplotlib.use('Agg')  # to avoid using Xserver
import matplotlib.pyplot as plt
import torch

from programl import serialize_ops
from programl.task.dataflow.ggnn_batch_builder import DataflowGgnnBatchBuilder
from programl.task.dataflow import dataflow, vocabulary
from programl.proto import (
    checkpoint_pb2,
    epoch_pb2,
    program_graph_features_pb2,
    program_graph_pb2,
)
from programl.models.ggnn.ggnn import Ggnn
from programl.models.batch_results import BatchResults
from programl.models.base_graph_loader import BaseGraphLoader
from programl.graph.format.py.nx_format import ProgramGraphToNetworkX
import igraph as ig
from datetime import datetime
import logging
from copy import deepcopy
from networkx.drawing.nx_agraph import graphviz_layout
import pathlib
from typing import Any, Iterable, List, Tuple
from os import listdir
import random
random.seed(888)


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
    "random_test_size",
    0,
    "If > 0, randomly select this many graph evaluations.",
)
app.DEFINE_float(
    "max_removed_edges_ratio",
    -1,
    "If > -1, limit the max number of removed edges.",
)
app.DEFINE_boolean(
    "filter_adjacant_nodes",
    False,
    "If set, filter out nodes that are too far from source.",
)
app.DEFINE_float("target_vocab_cumfreq", 1.0,
                 "The target cumulative frequency that.")
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
app.DEFINE_integer(
    "instance_id",
    None,
    "ID for this instance (for concurrency)."
)
app.DEFINE_integer(
    "num_instances",
    None,
    "Total number of instances to run (for concurrency)."
)
app.DEFINE_string(
    "task",
    "datadep",
    "Specify what task to test against."
)
app.DEFINE_boolean(
    "debug",
    False,
    "Whether to stop encountering exceptions."
)
FLAGS = app.FLAGS

ATTR_ACC_ASC_ORDER_TASKS = {
    "datadep",
}
ATTR_ACC_DES_ORDER_TASKS = {
    "reachability",
    "domtree",
    "liveness",
}


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


class TooManyEdgesRemovedError(Exception):
    pass


class NoQualifiedOutNodeError(Exception):
    pass


def RemoveCyclesFromGraph(
    networkx_graph: nx.DiGraph,
    method: str = "eades",
) -> nx.DiGraph:
    # This is a sufficint solution, with optimality guarantee if "ip" method
    # is seleted (but it will be very slow on large graph). A heuristic-based
    # method "eades" is also provided, with fast speed but only upper bound of
    # the number of removed edges of "|E|/2 - |V|/6".
    print("Removing cycles...")
    igraph_graph = ig.Graph.from_networkx(networkx_graph)
    edges_to_remove = ig.Graph.feedback_arc_set(igraph_graph, method=method)
    edge_list = igraph_graph.get_edgelist()
    for edge_id in edges_to_remove:
        tail, head = edge_list[edge_id]
        networkx_graph.remove_edge(u=tail, v=head)
    return networkx_graph


def FilterDistantNodes(
    nodes_out: List[int],
    target_node_id: int,
    graph: program_graph_pb2.ProgramGraph,
) -> List[int]:
    filtered_nodes_out = []
    networkx_graph = ProgramGraphToNetworkX(graph)
    for source_node_id in nodes_out:
        dist = nx.algorithms.shortest_paths.generic.shortest_path_length(
            networkx_graph,
            source=source_node_id,
            target=target_node_id,
        )
        if dist > 1:
            filtered_nodes_out.append(source_node_id)
    return filtered_nodes_out


def CalculateInterpolationOrderFromGraph(
    graph: program_graph_pb2.ProgramGraph,
    reverse: bool = False,
    max_removed_edges_ratio: float = -1,
) -> List[int]:
    # This function returns the (topological) order of nodes to evaluate
    # for interpolations in IG
    networkx_graph = ProgramGraphToNetworkX(graph)
    is_acyclic = nx.algorithms.dag.is_directed_acyclic_graph(networkx_graph)
    if is_acyclic:
        ordered_nodes = nx.topological_sort(networkx_graph)
        if reverse:
            ordered_nodes = list(ordered_nodes)
            ordered_nodes.reverse()
            return ordered_nodes, networkx_graph
        else:
            return list(ordered_nodes), networkx_graph
    else:
        # Cycle(s) detected and we need to remove them now
        if max_removed_edges_ratio != -1:
            original_num_edges = len(networkx_graph.edges)
        acyclic_networkx_graph = RemoveCyclesFromGraph(networkx_graph)
        if max_removed_edges_ratio != -1:
            trimmed_num_edges = len(acyclic_networkx_graph.edges)
            num_removed_edges = original_num_edges - trimmed_num_edges
            print("Total edges: %d | Removed %d edges." %
                  (original_num_edges, num_removed_edges))
            if (num_removed_edges / original_num_edges) > max_removed_edges_ratio:
                raise TooManyEdgesRemovedError

        # Sanity check, only return the graph if it is acyclic
        is_acyclic = nx.algorithms.dag.is_directed_acyclic_graph(
            acyclic_networkx_graph)
        if not is_acyclic:
            raise CycleInGraphError
        else:
            ordered_nodes = nx.topological_sort(acyclic_networkx_graph)
            if reverse:
                ordered_nodes = list(ordered_nodes)
                ordered_nodes.reverse()
                return ordered_nodes, acyclic_networkx_graph
            else:
                return list(ordered_nodes), acyclic_networkx_graph


def GenerateInterpolationOrderFromGraph(
    features_list_path: pathlib.Path,
    features_list_index: int,
    ds_path: str,
    max_removed_edges_ratio: float,
) -> Tuple[List[int], nx.DiGraph]:
    path = pathlib.Path(ds_path)

    features_list = pbutil.FromFile(
        features_list_path,
        program_graph_features_pb2.ProgramGraphFeaturesList(),
    )
    features = features_list.graph[features_list_index]

    graph_name = features_list_path.name[: -
                                         len(".ProgramGraphFeaturesList.pb")]
    graph = pbutil.FromFile(
        path / "graphs" / f"{graph_name}.ProgramGraph.pb",
        program_graph_pb2.ProgramGraph(),
    )

    if FLAGS.max_vis_graph_complexity != 0:
        if len(graph.node) > FLAGS.max_vis_graph_complexity:
            raise TooComplexGraphError
        if len(graph.edge) > FLAGS.max_vis_graph_complexity * 2:
            raise TooComplexGraphError

    # First, we need to fix empty node features
    graph = FixEmptyNodeFeatures(graph, features)

    interpolation_order, acyclic_networkx_graph = CalculateInterpolationOrderFromGraph(
        graph,
        max_removed_edges_ratio=max_removed_edges_ratio,
    )
    return interpolation_order, acyclic_networkx_graph


def TestOne(
    features_list_path: pathlib.Path,
    features_list_index: int,
    checkpoint_path: pathlib.Path,
    ds_path: str,
    run_ig: bool,
    dep_guided_ig: bool,
    all_nodes_out: bool,
    reverse: bool,
    filter_adjacant_nodes: bool,
    accumulate_gradients: bool,
    interpolation_order: List[int],
    acyclic_networkx_graph: nx.DiGraph,
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

    graph_name = features_list_path.name[: -
                                         len(".ProgramGraphFeaturesList.pb")]
    graph = pbutil.FromFile(
        path / "graphs" / f"{graph_name}.ProgramGraph.pb",
        program_graph_pb2.ProgramGraph(),
    )

    # First, we need to fix empty node features
    graph = FixEmptyNodeFeatures(graph, features)

    interpolation_order = deepcopy(interpolation_order)

    if run_ig:  # we can also compute accuracies for standard IG
        if reverse:
            interpolation_order.reverse()
        if not dep_guided_ig:
            interpolation_order = None
    else:
        interpolation_order = None

    acyclic_networkx_graph = deepcopy(acyclic_networkx_graph)

    root_nodes = []
    if all_nodes_out:
        nodes_out = []
        for i in range(len(graph.node)):
            if features.node_features.feature_list["data_flow_root_node"].feature[i].int64_list.value == [1]:
                root_nodes.append(i)
            if features.node_features.feature_list["data_flow_value"].feature[i].int64_list.value == [1]:
                nodes_out.append(i)

        # Filter nodes that are not suitable for evaluations (too far).
        if filter_adjacant_nodes:
            nodes_out = FilterDistantNodes(nodes_out, root_nodes[0], graph)
        if len(nodes_out) == 0:
            raise NoQualifiedOutNodeError
    else:
        for i in range(len(graph.node)):
            if features.node_features.feature_list["data_flow_root_node"].feature[i].int64_list.value == [1]:
                root_nodes.append(i)
    if len(root_nodes) > 1:
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
                accumulate_gradients=accumulate_gradients,
            )
            results_predicted_nodes.append(results)
        return AnnotateGraphWithBatchResultsForPredictedNodes(
            graph,
            features,
            results_predicted_nodes,
            nodes_out,
            run_ig,
            acyclic_networkx_graph,
        )
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


def CalculateAttributionAccuracyScore(
    graph: nx.DiGraph,
    attribution_order: List[int],
    source_node_id: int,
    target_node_id: int,
    use_all_paths: bool = False,
) -> float:
    if use_all_paths:
        all_paths = list(
            nx.algorithms.simple_paths.all_simple_paths(
                graph,
                source=source_node_id,
                target=target_node_id
            )
        )
    all_shortest_paths = list(
        nx.algorithms.shortest_paths.generic.all_shortest_paths(
            graph,
            source=source_node_id,
            target=target_node_id
        )
    )

    if use_all_paths:
        path_nodes_set = set([node for path in all_paths for node in path])
    shortest_path_nodes_set = set(
        [node for path in all_shortest_paths for node in path])

    if use_all_paths:
        path_score = 0.0
    shortest_path_score = 0.0
    for i in range(len(attribution_order)):
        attr_order = attribution_order[i]
        if use_all_paths:
            if i in path_nodes_set:
                path_score += 1 / (attr_order + 1)
        if i in shortest_path_nodes_set:
            shortest_path_score += 1 / (attr_order + 1)

    if use_all_paths:
        final_score = 0.5 * path_score + 0.5 * shortest_path_score
    else:
        final_score = shortest_path_score
    return final_score


def AnnotateGraphWithBatchResultsForPredictedNodes(
    base_graph: program_graph_pb2.ProgramGraph,
    features: program_graph_features_pb2.ProgramGraphFeatures,
    results_predicted_nodes: List[BatchResults],
    nodes_out: List[int],
    run_ig: bool,
    acyclic_networkx_graph: nx.DiGraph,
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
                node.features.feature["true_y"].int64_list.value.append(
                    true_y[0])
                node.features.feature["pred_y"].int64_list.value.append(
                    pred_y[0])
            else:
                node.features.feature["true_y"].int64_list.value.append(0)
                node.features.feature["pred_y"].int64_list.value.append(0)
            if run_ig:
                node.features.feature["attribution_order"].int64_list.value.append(
                    results.attributions[j])
                if features.node_features.feature_list["data_flow_root_node"].feature[j].int64_list.value == [1]:
                    if FLAGS.task in ATTR_ACC_ASC_ORDER_TASKS:
                        target_node_id = j
                    elif FLAGS.task in ATTR_ACC_DES_ORDER_TASKS:
                        source_node_id = j

        graph.features.feature["loss"].float_list.value.append(results.loss)
        graph.features.feature["accuracy"].float_list.value.append(
            results.accuracy)
        graph.features.feature["precision"].float_list.value.append(
            results.precision)
        graph.features.feature["recall"].float_list.value.append(
            results.recall)
        graph.features.feature["f1"].float_list.value.append(results.f1)
        graph.features.feature["confusion_matrix"].int64_list.value[:] = np.hstack(
            results.confusion_matrix
        )

        if run_ig:
            if FLAGS.task in ATTR_ACC_ASC_ORDER_TASKS:
                source_node_id = nodes_out[i]
            elif FLAGS.task in ATTR_ACC_DES_ORDER_TASKS:
                target_node_id = nodes_out[i]
            try:
                attribution_acc_score = CalculateAttributionAccuracyScore(
                    acyclic_networkx_graph,
                    results.attributions,
                    source_node_id,
                    target_node_id,
                )
            except nx.exception.NetworkXNoPath:
                print("No feasible from source %d to target %d was found!" %
                      (source_node_id, target_node_id))
                graph.features.feature["attribution_accuracy"].float_list.value.append(
                    -1.0)
                continue
            print("Feasible from source %d to target %d was found." %
                  (source_node_id, target_node_id))
            graph.features.feature["attribution_accuracy"].float_list.value.append(
                attribution_acc_score)

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
        node.features.feature["correct"].int64_list.value.append(
            true_y[i] == pred_y[i])
        if run_ig:
            node.features.feature["attribution_order"].int64_list.value.append(
                results.attributions[i])

    graph.features.feature["loss"].float_list.value.append(results.loss)
    graph.features.feature["accuracy"].float_list.value.append(
        results.accuracy)
    graph.features.feature["precision"].float_list.value.append(
        results.precision)
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
    run_ig=False,
    dep_guided_ig=False,
    all_nodes_out=False,
    reverse=False,
    filter_adjacant_nodes=False,
    accumulate_gradients=True,
    interpolation_order=None,
    acyclic_networkx_graph=None,
):
    if all_nodes_out:
        graphs = TestOne(
            features_list_path=pathlib.Path(graph_path),
            features_list_index=int(graph_idx),
            checkpoint_path=pathlib.Path(ds_path + model_path),
            ds_path=ds_path,
            run_ig=run_ig,
            dep_guided_ig=dep_guided_ig,
            all_nodes_out=all_nodes_out,
            reverse=reverse,
            filter_adjacant_nodes=filter_adjacant_nodes,
            accumulate_gradients=accumulate_gradients,
            interpolation_order=interpolation_order,
            acyclic_networkx_graph=acyclic_networkx_graph,
        )
        torch.cuda.empty_cache()
        return graphs
    else:
        graph = TestOne(
            features_list_path=pathlib.Path(graph_path),
            features_list_index=int(graph_idx),
            checkpoint_path=pathlib.Path(ds_path + model_path),
            ds_path=ds_path,
            run_ig=run_ig,
            dep_guided_ig=dep_guided_ig,
            all_nodes_out=all_nodes_out,
            reverse=reverse,
            filter_adjacant_nodes=filter_adjacant_nodes,
            accumulate_gradients=accumulate_gradients,
            interpolation_order=interpolation_order,
            acyclic_networkx_graph=acyclic_networkx_graph,
        )
        torch.cuda.empty_cache()
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

    scores = []

    for i in range(len(graphs)):
        graph = graphs[i]
        if graph.features.feature["attribution_accuracy"].float_list.value[0] == -1.0:
            continue

        save_graph_path = ds_path + '/vis_res/' + graph_fname + \
            ".AttributedProgramGraphFeaturesList.%s.%d.pb" % (suffix, i)
        if save_graph:
            print("Saving annotated graph to %s..." % save_graph_path)
            serialize_ops.save_graphs(save_graph_path, [graph])

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
        nx.draw(networkx_graph, pos=pos, labels=labels,
                node_size=500, node_color=color)

        if save_vis:
            save_img_path = ds_path + '/vis_res/' + graph_fname + \
                ".AttributedProgramGraph.%s.%d.png" % (suffix, i)
            print("Saving visualization of annotated graph to %s..." %
                  save_img_path)
            attr_acc_score = graph.features.feature["attribution_accuracy"].float_list.value[0]
            plt.text(x=20, y=20, s="Attr acc: %f" % attr_acc_score)
            plt.show()
            plt.savefig(save_img_path, format="PNG")
            plt.clf()

            scores.append(attr_acc_score)

    return scores


def Main():
    """Main entry point."""
    dataflow.PatchWarnings()

    instance_id = FLAGS.instance_id - 1  # due to Shell for loop convention

    # Handle all logging stuff
    now = datetime.now()
    ts_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    fmt = logging.Formatter("%(asctime)s | %(message)s")

    log_filepath = FLAGS.ds_path + \
        "/exp_log/batch_exp_%s_res_%d_%s.log" % (
            FLAGS.task, instance_id, ts_string)
    logger = logging.getLogger("logger")
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(log_filepath, mode="w")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    logger.debug("Log file being written into at %s" % log_filepath)

    if FLAGS.batch:
        graphs_dir = FLAGS.ds_path + '/labels/%s/' % FLAGS.task
        # sort the list to ensure the stability of concurrency
        graph_fnames = sorted(listdir(graphs_dir))
        if FLAGS.random_test_size != 0:
            random.shuffle(graph_fnames)
            success_count = 0

        variant_ranks = {
            "STANDARD_IG": [],
            "ASCENDING_DEPENDENCY_GUIDED_IG": [],
            "UNACCUMULATED_ASCENDING_DEPENDENCY_GUIDED_IG": [],
            "DESCENDING_DEPENDENCY_GUIDED_IG": [],
        }
        logger.info("GRAPH_NAME,STANDARD_IG,ASCENDING_DEPENDENCY_GUIDED_IG,UNACCUMULATED_ASCENDING_DEPENDENCY_GUIDED_IG,DESCENDING_DEPENDENCY_GUIDED_IG")
        for i in range(len(graph_fnames)):
            graph_fname = graph_fnames[i]
            if i % FLAGS.num_instances != instance_id:
                continue
            try:
                original_graph_fname = graph_fname[: -
                                                   len(".ProgramGraphFeaturesList.pb")].split('/')[-1]
                print("Processing graph file: %s..." % graph_fname)
                graph_path = graphs_dir + graph_fname
                try:
                    interpolation_order, acyclic_networkx_graph = GenerateInterpolationOrderFromGraph(
                        features_list_path=pathlib.Path(graph_path),
                        features_list_index=int('-1'),
                        ds_path=FLAGS.ds_path,
                        max_removed_edges_ratio=FLAGS.max_removed_edges_ratio,
                    )
                    graph_std_ig = TestOneGraph(
                        FLAGS.ds_path,
                        FLAGS.model,
                        graph_path,
                        '-1',
                        run_ig=FLAGS.ig,
                        dep_guided_ig=False,
                        all_nodes_out=FLAGS.only_pred_y,
                        filter_adjacant_nodes=FLAGS.filter_adjacant_nodes,
                        interpolation_order=interpolation_order,
                        acyclic_networkx_graph=acyclic_networkx_graph,
                    )
                    graph_dep_guided_ig = TestOneGraph(
                        FLAGS.ds_path,
                        FLAGS.model,
                        graph_path,
                        '-1',
                        run_ig=FLAGS.ig,
                        dep_guided_ig=True,
                        all_nodes_out=FLAGS.only_pred_y,
                        reverse=False,
                        filter_adjacant_nodes=FLAGS.filter_adjacant_nodes,
                        accumulate_gradients=True,
                        interpolation_order=interpolation_order,
                        acyclic_networkx_graph=acyclic_networkx_graph,
                    )
                    graph_dep_guided_ig_unaccumulated = TestOneGraph(
                        FLAGS.ds_path,
                        FLAGS.model,
                        graph_path,
                        '-1',
                        run_ig=FLAGS.ig,
                        dep_guided_ig=True,
                        all_nodes_out=FLAGS.only_pred_y,
                        reverse=False,
                        filter_adjacant_nodes=FLAGS.filter_adjacant_nodes,
                        accumulate_gradients=False,
                        interpolation_order=interpolation_order,
                        acyclic_networkx_graph=acyclic_networkx_graph,
                    )
                    graph_reverse_dep_guided_ig = TestOneGraph(
                        FLAGS.ds_path,
                        FLAGS.model,
                        graph_path,
                        '-1',
                        run_ig=FLAGS.ig,
                        dep_guided_ig=True,
                        all_nodes_out=FLAGS.only_pred_y,
                        reverse=True,
                        filter_adjacant_nodes=FLAGS.filter_adjacant_nodes,
                        accumulate_gradients=True,
                        interpolation_order=interpolation_order,
                        acyclic_networkx_graph=acyclic_networkx_graph,
                    )
                    print("Acyclic graph found and loaded.")
                except TooComplexGraphError:
                    print("Skipping graph %s due to exceeding number of nodes..." %
                          original_graph_fname)
                    continue
                except CycleInGraphError:
                    print("Skipping graph %s due to presence of graph cycle(s)..." %
                          original_graph_fname)
                    continue
                except TooManyEdgesRemovedError:
                    print("Skipping graph %s due to exceeding number of removed edges..." %
                          original_graph_fname)
                    continue
                except TooManyRootNodesError:
                    print("Skipping graph %s due to exceeding number of root nodes..." %
                          original_graph_fname)
                    continue
                except NoQualifiedOutNodeError:
                    print("Skipping graph %s due to no out node found..." %
                          original_graph_fname)
                    continue

                if FLAGS.ig and FLAGS.dep_guided_ig:
                    attr_acc_std_ig = DrawAndSaveGraph(
                        graph_std_ig, FLAGS.ds_path,
                        original_graph_fname, save_graph=FLAGS.save_graph,
                        save_vis=FLAGS.save_vis, suffix='std_ig'
                    )
                    attr_acc_dep_guided_ig = DrawAndSaveGraph(
                        graph_dep_guided_ig, FLAGS.ds_path,
                        original_graph_fname, save_graph=FLAGS.save_graph,
                        save_vis=FLAGS.save_vis, suffix='dep_guided_ig'
                    )
                    attr_acc_dep_guided_ig_unaccumulated = DrawAndSaveGraph(
                        graph_dep_guided_ig_unaccumulated, FLAGS.ds_path,
                        original_graph_fname, save_graph=FLAGS.save_graph,
                        save_vis=FLAGS.save_vis, suffix='dep_guided_ig_unaccumulated'
                    )
                    attr_acc_reverse_dep_guided_ig = DrawAndSaveGraph(
                        graph_dep_guided_ig, FLAGS.ds_path,
                        original_graph_fname, save_graph=FLAGS.save_graph,
                        save_vis=FLAGS.save_vis, suffix='reverse_dep_guided_ig'
                    )

                    for attr_acc_std_ig, attr_acc_dep_guided_ig, attr_acc_dep_guided_ig_unaccumulated, attr_acc_reverse_dep_guided_ig in zip(attr_acc_std_ig, attr_acc_dep_guided_ig, attr_acc_dep_guided_ig_unaccumulated, attr_acc_reverse_dep_guided_ig):
                        # sorted_acc_scores = sorted([
                        #     attr_acc_std_ig,
                        #     attr_acc_dep_guided_ig,
                        #     attr_acc_dep_guided_ig_unaccumulated,
                        #     attr_acc_reverse_dep_guided_ig,
                        # ])
                        # variant_rank = list(map(lambda x: sorted_acc_scores.index(x), [
                        #     attr_acc_std_ig,
                        #     attr_acc_dep_guided_ig,
                        #     attr_acc_dep_guided_ig_unaccumulated,
                        #     attr_acc_reverse_dep_guided_ig
                        # ]))
                        # variant_ranks["STANDARD_IG"].append(variant_rank[0])
                        # variant_ranks["ASCENDING_DEPENDENCY_GUIDED_IG"].append(
                        #     variant_rank[1])
                        # variant_ranks["UNACCUMULATED_ASCENDING_DEPENDENCY_GUIDED_IG"].append(
                        #     variant_rank[2])
                        # variant_ranks["DESCENDING_DEPENDENCY_GUIDED_IG"].append(
                        #     variant_rank[3])

                        # rank_str = ""
                        # for variant_name, ranks in variant_ranks.items():
                        #     mean_rank = sum(ranks) / len(ranks)
                        #     rank_str += "|%s: %f|" % (variant_name, mean_rank)
                        # rank_str = '[' + rank_str + ']'

                        logger.info("%s,%f,%f,%f,%f" % (
                            graph_fname,
                            attr_acc_std_ig,
                            attr_acc_dep_guided_ig,
                            attr_acc_dep_guided_ig_unaccumulated,
                            attr_acc_reverse_dep_guided_ig,
                        ))
                if FLAGS.random_test_size != 0:
                    success_count += 1
                    print("Successfully finished %d graphs." % success_count)
                    if success_count > FLAGS.random_test_size:
                        print("Finished all graphs.")
                        exit()

            except Exception as err:
                if FLAGS.debug:
                    raise err
                else:
                    print("Error testing %s -- %s" % (graph_fname, str(err)))
                    continue
    else:
        features_list_path, features_list_index = FLAGS.input.split(":")
        graph_fname = features_list_path[: -
                                         len(".ProgramGraphFeaturesList.pb")].split('/')[-1]
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
                filter_adjacant_nodes=FLAGS.filter_adjacant_nodes,
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
                reverse=False,
                filter_adjacant_nodes=FLAGS.filter_adjacant_nodes,
                accumulate_gradients=True,
            )
            graph_dep_guided_ig_unaccumulated = TestOneGraph(
                FLAGS.ds_path,
                FLAGS.model,
                FLAGS.ds_path + features_list_path,
                features_list_index,
                FLAGS.max_vis_graph_complexity,
                run_ig=FLAGS.ig,
                dep_guided_ig=True,
                all_nodes_out=FLAGS.only_pred_y,
                reverse=False,
                filter_adjacant_nodes=FLAGS.filter_adjacant_nodes,
                accumulate_gradients=False,
            )
            graph_reverse_dep_guided_ig = TestOneGraph(
                FLAGS.ds_path,
                FLAGS.model,
                FLAGS.ds_path + features_list_path,
                features_list_index,
                FLAGS.max_vis_graph_complexity,
                run_ig=FLAGS.ig,
                dep_guided_ig=True,
                all_nodes_out=FLAGS.only_pred_y,
                reverse=True,
                filter_adjacant_nodes=FLAGS.filter_adjacant_nodes,
                accumulate_gradients=True,
            )
        except TooComplexGraphError:
            print("Skipping graph %s due to exceeding number of nodes..." %
                  graph_fname)
            exit()
        except CycleInGraphError:
            print("Skipping graph %s due to presence of graph cycle(s)..." %
                  graph_fname)
            exit()
        except TooManyEdgesRemovedError:
            print(
                "Skipping graph %s due to exceeding number of removed edges..." % graph_fname)
            exit()
        except TooManyRootNodesError:
            print(
                "Skipping graph %s due to exceeding number of root nodes..." % graph_fname)
            exit()
        except NoQualifiedOutNodeError:
            print("Skipping graph %s due to no out node found..." % graph_fname)
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
            DrawAndSaveGraph(
                graph_dep_guided_ig, FLAGS.ds_path,
                graph_fname, save_graph=FLAGS.save_graph,
                save_vis=FLAGS.save_vis, suffix='dep_guided_ig_unaccumulated'
            )
            DrawAndSaveGraph(
                graph_reverse_dep_guided_ig, FLAGS.ds_path,
                graph_fname, save_graph=FLAGS.save_graph,
                save_vis=FLAGS.save_vis, suffix='reverse_dep_guided_ig'
            )


if __name__ == "__main__":
    app.Run(Main)
