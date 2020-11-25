// Copyright 2019-2020 the ProGraML authors.
//
// Contact Chris Cummins <chrisc.101@gmail.com>.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <iostream>

#include "boost/filesystem.hpp"
#include "labm8/cpp/app.h"
#include "tasks/classifyapp/dataset/poj104.h"

namespace fs = boost::filesystem;

const char* usage = R"(Download and create the POJ-104 dataset.

The POJ-104 dataset contains 52000 C++ programs implementing 104 different
algorithms with 500 examples of each.

The dataset is from:

  Mou, L., Li, G., Zhang, L., Wang, T., & Jin, Z. (2016). Convolutional Neural
  Networks over Tree Structures for Programming Language Processing. AAAI.

And is available at:

  https://sites.google.com/site/treebasedcnn/

This program creates the 52,000 source files, LLVM-IR files, and program graphs,
divided intro training, validation, and test data.)";

DEFINE_string(url,
              "https://drive.google.com/u/0/"
              "uc?id=0B2i-vWnOu7MxVlJwQXN6eVNONUU&export=download",
              "The URL of the author-provided archive.tar.gz file.");
DEFINE_string(path, "/tmp/programl/poj104", "The directory to write generated files to.");

int main(int argc, char** argv) {
  labm8::InitApp(&argc, &argv);
  if (argc > 1) {
    std::cerr << "fatal: Unrecognized arguments" << std::endl;
    return 4;
  }
  programl::task::classifyapp::CreatePoj104Dataset(FLAGS_url, fs::path(FLAGS_path));
  return 0;
}
