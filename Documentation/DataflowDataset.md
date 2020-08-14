# Dataflow Dataset

The data flow dataset contains LLVM-IRs taken from a wide range of
projects and source programming languages, and includes labels for
several compiler data flows. We also include the logs for the machine
learning jobs which produced our published experimental results.


## Download links

| Dataset  | Download Size | Uncompressed Size | Description |
| ------------- | ------------- | ------------- | ------------- |
| [`llvm_ir_20.06.01.tar.bz2`](https://www.dropbox.com/s/cw25t4cls6otel5/llvm_ir_20.06.01.tar.bz2)  |  1.1GB  |  14GB  | 469,086 real-world LLVM-IR files taken from a variety of C, C++, Swift, Fortran, and OpenCL projects.  |
| [`graphs_20.06.01.tar.bz2`](https://www.dropbox.com/s/f7j2qbo6hobckqy/graphs_20.06.01.tar.bz2)  |  3.5GB  |  67GB  |  ProGraML graphs for the above LLVM-IRs, divided into 3:1:1 training, validation, and test splits. |
| [`labels_reachability_20.06.01.tar.bz2`](https://www.dropbox.com/s/ha0lp343an336fp/labels_reachability_20.06.01.tar.bz2)  |  81MB  |  25GB  |  Reachability analysis labels for the graphs.  |
| [`labels_domtree_20.06.01.tar.bz2`](https://www.dropbox.com/s/cl4kuak4kds9yuc/labels_domtree_20.06.01.tar.bz2)  |  66MB  |  18GB  |  Dominator analysis labels for the graphs.  |
| [`labels_datadep_20.06.01.tar.bz2`](https://www.dropbox.com/s/yxqzmp4vvzycxje/labels_datadep_20.06.01.tar.bz2)  |  67MB  |  26GB  |  Data dependency analysis labels for the graphs.  |
| [`labels_liveness_20.06.01.tar.bz2`](https://www.dropbox.com/s/qvhekvovx6n1ino/labels_liveness_20.06.01.tar.bz2)  |  119MB  |  24GB  |  Live-out variable analysis labels for the graphs.  |
| [`labels_subexpressions_20.06.01.tar.bz2`](https://www.dropbox.com/s/a8tkugnu6mq64kq/labels_subexpressions_20.06.01.tar.bz2)  |  69MB  |  26GB  |  Common subexpression analysis labels for the graphs.  |
| [`dataflow_logs_20.06.01.tar.bz2`](https://www.dropbox.com/s/o3qopbyhf2p5pf2/dataflow_logs_20.06.01.tar.bz2)  |  254MB  |  412MB  |  Configs, logs, and trained models for ProGraML/inst2vec/CDFG.  |


## Directory Layout

The uncompressed dataset uses the following layout:

* `labels/`
    * Directory containing machine learning features and labels for
      programs for compiler data flow analyses.
    * `labels/<analysis>/<source>.<id>.<lang>.ProgramFeaturesList.pb`
        * A
          [ProgramFeaturesList](/programl/proto/program_graph_features.proto)
          protocol buffer containing a list of features resulting from
          running a data flow analysis on a program.
* `graphs/`
    * Directory containing ProGraML representations of LLVM IRs.
    * `graphs/<source>.<id>.<lang>.ProgramGraph.pb`
        * A [ProgramGraph](/programl/proto/program_graph.proto)
          protocol buffer of an LLVM IR in the ProGraML
          representation.
* `ll/`
    * Directory containing LLVM-IR files.
    * `ir/<source>.<id>.<lang>.ll`
        * An LLVM IR in text format, as produced by `clang -emit-llvm
          -S` or equivalent.
* `test/`
    * A directory containing symlinks to graphs in the `graphs/`
      directory, indicating which graphs should be used as part of the
      test set.
* `train/`
    * A directory containing symlinks to graphs in the `graphs/`
      directory, indicating which graphs should be used as part of the
      training set.
* `val/`
    * A directory containing symlinks to graphs in the `graphs/`
      directory, indicating which graphs should be used as part of the
      validation set.
* `vocal/`
    * Directory containing vocabulary files.
    * `vocab/<type>.csv`
      * A vocabulary file, which lists unique node texts, their
        frequency in the dataset, and the cumulative proportion of
        total unique node texts that is covered.


After running the dataflow experiments, log files are produced in this
directory with the structure:

* `logs/<model>/<analysis>/<run_id>`
    * Logs recording the performance of a model running on a specific
      analysis. `<run_id>` is a timestamp used to distinguish multiple
      runs of the same model on the same analysis.
    * `logs/<model>/<analysis>/<run_id>/flags.txt`
        * A text file listing the command-line flags that were set
          during this run, either explicitly, or through default
          values.
    * `logs/<model>/<analysis>/<run_id>/build_info.json`
        * A JSON file describing the state of the code base that was
          used. This, along with the `flags.txt`, fully describes the
          experimental setup.
    * `logs/<model>/<analysis>/<run_id>/checkpoints/`
        * Directory containing model checkpoints, which are restorable
          snapshots of model state.
        * `logs/<model>/<analysis>/<run_id>/checkpoints/<epoch>.Checkpoint.pb`
            * A [Checkpoint](/programl/proto/checkpoint.proto)
              protocol buffer recording a snapshot of the model state
              at the end of the given epoch.
    * `logs/<model>/<analysis>/<run_id>/epochs/`
        * `logs/<model>/<analysis>/<run_id>/epochs/<epoch>.EpochList.pbtxt`
            * An [EpochList](/programl/proto/epoch.proto) protocol
              buffer recording the performance of the model during
              this epoch.
    * `logs/<model>/<analysis>/<run_id>/graph_loader/`
        * A directory containing graph loader log files.
        * `logs/<model>/<analysis>/<run_id>/graph_loader/<epoch>.txt`
            * A text file containing a list of the graphs that were
              loaded in a given epoch, in the order they were read.


### File Types

To save disk space, most of the protocol buffers are stored in binary
wire format, indicated by the `.pb` file extension. The
[pbq](cmd/pbq.txt) program can used to decode binary protocol buffers
into a human-readable text format. To do so, you must specify the type
of the message, indicated using a `.<type>.pb` suffix on the
filename. For example, to decode the ProgramGraph protocol buffer
`graphs/foo.c.ProgramGraph.pb`, run:

```sh
$ pbq ProgramGraph --stdin_fmt=pb < graphs/foo.c.ProgramGraph.pb
```


## Attribution

If you use this dataset, please cite:

```
@article{cummins2020a,
  title={ProGraML: Graph-based Deep Learning for Program Optimization and Analysis},
  author={Cummins, Chris and Fisches, Zacharias and Ben-Nun, Tal and Hoefler, Torsten and Leather, Hugh},
  journal={arXiv:2003.10536v1},
  year={2020}
}
```
