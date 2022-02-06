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

cmake_minimum_required(VERSION 3.10)

file(READ "version.txt" version)
string(STRIP ${version} version)

project(ProGraML VERSION ${version})

# NOTE(cummins): LLVM 3.8 is not compatible with C++17 or later.
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(Protobuf REQUIRED)
# Add the project root to the "#import" search path.
set(Protobuf_IMPORT_DIRS ${PROJECT_SOURCE_DIR})

add_subdirectory(programl)
