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
#include "programl/task/dataflow/dataset/parallel_file_map.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "boost/filesystem.hpp"
#include "labm8/cpp/app.h"
#include "labm8/cpp/logging.h"
#include "labm8/cpp/strutil.h"

#include "absl/strings/str_format.h"

namespace fs = boost::filesystem;
using std::vector;

DEFINE_int32(limit, 0,
             "If --limit > 0, limit the number of input graphs processed to "
             "this number.");

namespace programl {
namespace task {
namespace dataflow {

vector<fs::path> EnumerateProgramGraphFiles(const fs::path& root) {
  vector<fs::path> files;
  for (auto it : fs::directory_iterator(root)) {
    if (labm8::HasSuffixString(it.path().string(), ".ProgramGraph.pb")) {
      files.push_back(it.path());
    }
  }

  // Randomize the order of files to crudely load balancing a
  // bunch of parallel workers iterating through this list in order
  // as the there is a high variance in the size / complexity of files.
  std::srand(unsigned(std::time(0)));
  std::random_shuffle(files.begin(), files.end());
  return files;
}

}  // namespace dataflow
}  // namespace task
}  // namespace programl
