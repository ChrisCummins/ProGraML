# Neural Code Comprehension: A Learnable Representation of Code Semantics

```ncc``` (Neural Code Comprehension) is a general Machine Learning technique to learn semantics from raw code in virtually any programming language. It relies on ```inst2vec```, an embedding space and graph representation of LLVM IR statements and their context.

![ncc_scheme](figures/overview.png)

This repository contains the code used in [[paper](http://arxiv.org/abs/1806.07336)]:
> Neural Code Comprehension: A Learnable Representation of Code Semantics, Tal Ben-Nun, Alice Shoshana Jakobovits, Torsten Hoefler

Please cite as:
```bibtex
@incollection{ncc,
title = {Neural Code Comprehension: A Learnable Representation of Code Semantics},
author = {Ben-Nun, Tal and Jakobovits, Alice Shoshana and Hoefler, Torsten},
booktitle = {Advances in Neural Information Processing Systems 31},
editor = {S. Bengio and H. Wallach and H. Larochelle and K. Grauman and N. Cesa-Bianchi and R. Garnett},
pages = {3588--3600},
year = {2018},
publisher = {Curran Associates, Inc.},
url = {http://papers.nips.cc/paper/7617-neural-code-comprehension-a-learnable-representation-of-code-semantics.pdf}
}
```

## Code

### Requirements

For training ```inst2vec``` embeddings:
* GNU / Linux or Mac OS
* Python (3.6.5)
  * tensorflow (1.7.0) or preferably: tensorflow-gpu (1.7.0)
  * networkx (2.1)
  * scipy (1.1.0)
  * absl-py (0.2.2)
  * jinja2 (2.10)
  * bokeh (0.12.16)
  * umap (0.1.1)
  * sklearn (0.0)
  * wget (3.2)

Additionally, for training ```ncc``` models:
* GNU / Linux or Mac OS
* Python (3.6.5)
  * labm8 (0.1.2)
  * keras (2.2.0)

### Running the code

#### 1. Training `inst2vec` embeddings

By default, `inst2vec` will be trained on publicly available code. Some additional datasets are available on demand and you may add them manually to the training data. For more information on how to do this as well as on the datasets in general, see [datasets](data/README.md).

```shell
$ python train_inst2vec.py --helpfull # to see the full list of options
$ python train_inst2vec.py \
>  # --context_width ... (default: 2)
>  # --data ... (default: data/, automatically generated one. You may provide your own)
```

Alternatively, you may skip this step and use [pre-trained embeddings](published_results/emb.p).

#### 2. Evaluating `inst2vec` embeddings

```shell
$ python train_inst2vec.py \
> --embeddings_file ... (path to the embeddings p-file to evaluate)
> --vocabulary_folder ... (path to the associated vocabulary folder)
```

#### 3. Training on tasks with ```ncc```

We provide the code for training three downstream tasks using the same neural architecture (```ncc```) and ```inst2vec``` embeddings.

**Algorithm classification**

Task: Classify applications into 104 classes given their raw code.
Code and classes provided by https://sites.google.com/site/treebasedcnn/ (see [Convolutional neural networks over tree structures for programming language processing](https://arxiv.org/abs/1409.5718))

Train:
```shell
$ python train_task_classifyapp.py --helpfull # to see the full list of options
$ python train_task_classifyapp.py
```

Alternatively, display results from a [pre-trained model](published_results).

**Optimal device mapping prediction**

Task: Predict the best-performing compute device (e.g., CPU, GPU)
Code and classes provided by https://github.com/ChrisCummins/paper-end2end-dl (see [End-to-end Deep Learning of Optimization Heuristics](https://hgpu.org/?p=17573))

Train:
```shell
$ python train_task_devmap.py --helpfull # to see the full list of options
$ python train_task_devmap.py
```

Alternatively, display results from a [pre-trained model](published_results).

**Optimal thread coarsening factor prediction**

Code and classes provided by https://github.com/ChrisCummins/paper-end2end-dl (see [End-to-end Deep Learning of Optimization Heuristics](https://hgpu.org/?p=17573))

Train:
```shell
$ python train_task_threadcoarsening.py --helpfull # to see the full list of options
$ python train_task_threadcoarsening.py
```

Alternatively, display results from a [pre-trained model](published_results).

## Contact

We would be thrilled if you used and built upon this work.
Contributions, comments, and issues are welcome!

## License

NCC is published under the New BSD license, see [LICENSE](LICENSE).

