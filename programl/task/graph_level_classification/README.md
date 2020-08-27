# Graph Level Classification

Two subtasks are of particular interest: classifyapp a.k.a. poj104 and devmap a.k.a. heterogeneous device mapping.

## Quickstart
`python run.py --help` will print this help
```
Usage:
   run.py [options]

Options:
    -h --help                       Show this screen.
    --data_dir DATA_DIR             Directory(*) to of dataset. (*)=relative to repository root ProGraML/.
                                        Will overwrite the per-dataset defaults if provided.

    --log_dir LOG_DIR               Directory(*) to store logfiles and trained models relative to repository dir.
                                        [default: programl/task/graph_level_classification/logs/unspecified]
    --model MODEL                   The model to run.
    --dataset DATASET               The dataset to use.
    --config CONFIG                 Path(*) to a config json dump with params.
    --config_json CONFIG_JSON       Config json with params.
    --restore CHECKPOINT            Path(*) to a model file to restore from.
    --skip_restore_config           Whether to skip restoring the config from CHECKPOINT.
    --test                          Test the model without training.
    --restore_by_pattern PATTERN    Restore newest model of this name from log_dir and
                                        continue training. (AULT specific!)
                                        PATTERN is a string that can be grep'ed for.
    --kfold                         Run kfold cross-validation iff kfold is set.
                                        Splits are currently dataset specific.
    --transfer MODEL                The model-class to transfer to.
                                    The args specified will be applied to the transferred model to the extend applicable, e.g.
                                        training params and Readout module specifications, but not to the transferred model trunk.
                                        However, we strongly recommend to make all trunk-parameters match, in order to be able
                                        to restore from transferred checkpoints without having to pass a matching config manually.
    --transfer_mode MODE            One of frozen, finetune (but not yet implemented) [default: frozen]
                                        Mode frozen also sets all dropout in the restored model to zero (the newly initialized
                                        readout function can have dropout nonetheless, depending on the config provided).
    --skip_save_every_epoch         Save latest model after every epoch (on a rolling basis).
```
Therefore, an exemplary command could look like this:
```
Reproduce the Transformer result for the rebuttal:

python run.py --model transformer_poj104 --dataset poj104 --data_dir ~/rebuttal_datasets/classifyapp/ --log_dir logs/classifyapp_logs/rebuttal_transformer_poj104/ --config_json="{'train_subset': [0, 100], 'batch_size': 48, 'max_num_nodes': 40000, 'num_epochs': 70, 'vocab_size': 2231, 'message_weight_sharing': 2, 'update_weight_sharing': 2, 'lr': 1e-4, 'gnn_layers': 10}" 
```
NB: You can pass a double quoted string of config options in json format, except that you may use single quotes (they will be parsed as double quotes to transform this almost-json format into valid json)

## How to reproduce results from the paper?

```
more run commands / another script that does it for us.
```

