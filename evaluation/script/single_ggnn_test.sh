#!/bin/bash

echo "Max number of graph nodes: $1"
echo "Max toleratable ratio of edge removal for removing cycles: $2"
echo "Task name: $3"

if [ $3 = "domtree" ]; then
    model_id="14"
else
    model_id="15"
fi

bazel run --verbose_failures //programl/task/dataflow:ggnn_test \
    -- --model=/logs/programl/$3/ddf_30/checkpoints/0$model_id.Checkpoint.pb \
    --ig -dep_guided_ig --save_vis --only_pred_y --batch --random_test_size 100 \
    --max_vis_graph_complexity $1 --max_removed_edges_ratio $2 --task $3 \
    --filter_adjacant_nodes --instance_id 1 --num_instances 1 --debug