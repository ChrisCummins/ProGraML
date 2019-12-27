# Agumented inst2vec embeddings.
#
# This extends the vocabulary used in the NeurIPS'18 paper to add three extra
# elements: !MAGIC, !IMMEDIATE, and !IDENTIFIER. These new vocabulary elements
# have been assigned random embedding vectors.
#
# Copyright 2019 the ProGraML authors.
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

filegroup(
    name = "dictionary",
    srcs = ["inst2vec_augmented_dictionary.pickle"],
    visibility = ["//deeplearning/ml4pl/graphs/llvm2graph:__subpackages__"],
)

filegroup(
    name = "embeddings",
    srcs = ["inst2vec_augmented_embeddings.pickle"],
    visibility = ["//deeplearning/ml4pl/graphs/llvm2graph:__subpackages__"],
)
