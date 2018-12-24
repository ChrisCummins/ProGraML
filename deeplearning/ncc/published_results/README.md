# Results

## Pre-trained [embeddings](emb.p)

**Data set and [vocabulary](https://polybox.ethz.ch/index.php/s/AWKd60qR63yViH8)**

Dataset | Files | LLVM IR lines | Vocabulary size | XFG stmt pairs
------------ | ------------- | ------------- | ------------- | -------------
Tensorflow | 2,492 | 16,943,893 | 220,554 | 260,250,973
AMD APP SDK | 123 | 1,304,669 | 4,146 | 45,081,359
BLAS | 300 | 280,782 | 566 | 283,856
NAS | 268 | 572,521 | 1,793 | 1,701,968
Parboil | 151 | 118,575 | 2,175 | 151,916
PolybenchGPU | 40 | 33,601 | 577 | 40,975
Rodinia | 92 | 103,296 | 3,861 | 266,354
SHOC | 112 | 399,287 | 3,381 | 12,096,508
COSMO | 161 | 152,127 | 2,344 | 2,338,153
Linux kernel | 1,988 | 2,544,245 | 136,545 | 5,271,179
OpenCV | 442 | 1,908,683 | 39,920 | 10,313,451
NVIDIA samples | 60 | 43,563 | 2,467 | 74,915
Synthetic | 17,801 | 26,045,547 | 113,763 | 303,054,685

Note: the "synthetic" data set is made up of:

Dataset | Files | LLVM IR lines | XFG stmt pairs
------------ | ------------- | ------------- | -------------
eigen | 1'301 | 19'796'291 | 254'997'306
gemm_eigen_sample | 500 | 3'208'180 | 46'027'593
gemm_simple_sample | 3'200 | 711'839 | 671'033
stencil_1d_sample | 3'200 | 395'056 | 233'552
stencil_2d_sample | 3'200 | 600'133 | 389'008
stencil_3d_sample | 3'200 | 728'849 | 433'985
stencil_mc4d_sample | 3'200 | 605'199 | 302'208

**Skip-Gram parameters**

Parameter | Value
------------ | -------------
Context width | x
x | x

**Training parameters**

Parameter | Value
------------ | -------------
Number epochs | x
x | x

## Pre-trained task models

**Algorithm classification [here](classifyapp/CLASSIFYAPP-94.83.h5)**

**Prediction Accuracy [%]**

Computing Platform |  [Grewe et al.](http://www.lancaster.ac.uk/staff/wangz3/publications/cgo_omp2ocl.pdf) | [DeepTune](https://chriscummins.cc/pub/2017-pact.pdf) | ncc / inst2vec
------------ | ------------- | ------------- | ------------- 
AMD Tahiti 7970 | 73.38 | 83.68 | 82.79
NVIDIA GTX 970 | 72.94 | 80.29 | 81.76

**Speedups**

Computing Platform | [Grewe et al.](http://www.lancaster.ac.uk/staff/wangz3/publications/cgo_omp2ocl.pdf) | [DeepTune](https://chriscummins.cc/pub/2017-pact.pdf) | ncc / inst2vec
------------ | ------------- | ------------- | ------------- 
AMD Tahiti 7970 | 2.91 | 3.34 | 3.42
NVIDIA GTX 970 | 1.26 | 1.41 | 1.39


**Optimal thread coarsening factor prediction [here](https://polybox.ethz.ch/index.php/s/F8FVQV1vig2KgPB)**

**Speedups**

Computing Platform | [Magni et al.](https://homepages.inf.ed.ac.uk/cdubach/papers/magni14pact.pdf) | [DeepTune](https://chriscummins.cc/pub/2017-pact.pdf) | [DeepTune-TL](https://chriscummins.cc/pub/2017-pact.pdf) | ncc / inst2vec
------------ | ------------- | ------------- | ------------- | ------------- 
AMD Radeon HD 5900 | 1.21 | 1.10 | 1.17 | 1.25
AMD Tahiti 7970 | 1.01 | 1.05 | 1.23 | 1.07
NVIDIA GTX 480 | 0.86 | 1.10 | 1.14 | 1.02
NVIDIA Tesla K20c | 0.94 | 0.99 | 0.93 | 1.03
