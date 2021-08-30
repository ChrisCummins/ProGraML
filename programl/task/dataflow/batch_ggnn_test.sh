#!/bin/bash

echo "Total number of instances to run: $1"

bazel run --verbose_failures programl/task/dataflow:ggnn_test \
    -- --model=/logs/programl/datadep/ddf_30/checkpoints/015.Checkpoint.pb \
    --ig -dep_guided_ig --save_vis --only_pred_y --batch \
    --max_vis_graph_complexity 30 --max_removed_edges_ratio 0.5 \
    --filter_adjacant_nodes