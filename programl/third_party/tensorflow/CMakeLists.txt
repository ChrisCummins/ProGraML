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

add_subdirectory(compiler/xla)

protobuf_generate_cpp(
    THIRD_PARTY_TENSORFLOW_PROTO_SRCS
    THIRD_PARTY_TENSORFLOW_PROTO_HDRS
    features.proto
    xla.proto
)

add_library(
    third_party_tensorflow_protos
    STATIC
    ${THIRD_PARTY_TENSORFLOW_PROTO_SRCS}
    ${THIRD_PARTY_TENSORFLOW_PROTO_HDRS}
    ${THIRD_PARTY_TENSORFLOW_COMPILER_XLA_PROTO_HDRS}
    ${THIRD_PARTY_TENSORFLOW_COMPILER_XLA_SERVICE_PROTO_HDRS}
)

set(THIRD_PARTY_TENSORFLOW_PROTO_SRCS ${THIRD_PARTY_TENSORFLOW_PROTO_SRCS} PARENT_SCOPE)
set(THIRD_PARTY_TENSORFLOW_PROTO_HDRS ${THIRD_PARTY_TENSORFLOW_PROTO_HDRS} PARENT_SCOPE)

target_include_directories(
    third_party_tensorflow_protos
    PUBLIC
    ${Protobuf_INCLUDE_DIRS}
    ${CMAKE_BINARY_DIR}
)
