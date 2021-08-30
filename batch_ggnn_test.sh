#!/bin/bash

echo "Total number of instances to run: $1"
echo "Max number of graph nodes: $2"
echo "Max toleratable ratio of edge removal for removing cycles: $3"

for instance_id in `seq 1 $1`
do
    echo "Starting instance with ID $instance_id..."
    nohup bazel run --verbose_failures programl/task/dataflow:ggnn_test \
        -- --model=/logs/programl/datadep/ddf_30/checkpoints/015.Checkpoint.pb \
        --ig -dep_guided_ig --save_vis --only_pred_y --batch \
        --max_vis_graph_complexity $2 --max_removed_edges_ratio $3 \
        --filter_adjacant_nodes --instance_id $instance_id --num_instances $1 > evaluation/log/nohup_datadep_exp_$2_$3_$1.log 2>&1 &
done