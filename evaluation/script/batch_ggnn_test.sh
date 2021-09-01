#!/bin/bash

echo "Total number of instances to run: $1"
echo "Max number of graph nodes: $2"
echo "Max toleratable ratio of edge removal for removing cycles: $3"
echo "Task name: $4"
echo "Total test size: $5"

per_instance_size=$(($5/$1))

echo "Per-instance test size: $per_instance_size"

if [ $4 = "domtree" ]; then
    model_id="14"
else
    model_id="15"
fi

for instance_id in `seq 1 $1`
do
    echo "Starting instance with ID $instance_id..."
    nohup bazel run --verbose_failures //programl/task/dataflow:ggnn_test \
        -- --model=/logs/programl/$4/ddf_30/checkpoints/0$model_id.Checkpoint.pb \
        --ig -dep_guided_ig --save_vis --only_pred_y --batch --random_test_size 20 \
        --max_vis_graph_complexity $2 --max_removed_edges_ratio $3 --task $4 \
        --filter_adjacant_nodes --instance_id $instance_id --num_instances $1 \
        > ../log/nohup_$4_exp_$2_$3_$1_$instance_id.log 2>&1 &
done