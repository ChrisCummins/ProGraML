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
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "absl/strings/str_format.h"
#include "boost/filesystem.hpp"
#include "labm8/cpp/app.h"
#include "labm8/cpp/fsutil.h"
#include "labm8/cpp/logging.h"
#include "labm8/cpp/strutil.h"
#include "programl/proto/util.pb.h"
#include "tasks/dataflow/dataset/parallel_file_map.h"

namespace fs = boost::filesystem;
using std::vector;

const char* usage = R"(Unpack IrList protos to Ir protos.)";

DEFINE_string(path, (labm8::fsutil::GetHomeDirectoryOrDie() / "programl/dataflow").string(),
              "The directory to write generated files to.");

namespace programl {
namespace task {
namespace dataflow {

inline string GetOutput(const fs::path& root, const string& nameStem, int index) {
  return absl::StrFormat("%s/ir/%s%d.Ir.pb", root.string(), nameStem, index);
}

void ProcessIrList(const fs::path& root, const fs::path& path) {
  const string baseName = path.string().substr(path.string().rfind("/") + 1);
  const string nameStem = baseName.substr(0, baseName.size() - labm8::StrLen("IrList.pb"));

  std::ifstream file(path.string());
  IrList irList;
  if (!irList.ParseFromIstream(&file)) {
    LOG(ERROR) << "Failed to parse: " << path.string();
    return;
  }

  // Write each Ir to its own file.
  for (int i = 0; i < irList.ir_size(); ++i) {
    const string outPath = absl::StrFormat("%s/ir/%s.%d.Ir.pb", root.string(), nameStem, i);
    std::ofstream out(outPath);
    irList.ir(i).SerializeToOstream(&out);
  }

  // Once we're done, delete the IrList.
  fs::remove(path);
}

vector<fs::path> EnumerateIrLists(const fs::path& root) {
  vector<fs::path> files;
  for (auto it : fs::directory_iterator(root)) {
    if (labm8::HasSuffixString(it.path().string(), ".IrList.pb")) {
      files.push_back(it.path());
    }
  }
  return files;
}

}  // namespace dataflow
}  // namespace task
}  // namespace programl

int main(int argc, char** argv) {
  labm8::InitApp(&argc, &argv);
  if (argc > 1) {
    std::cerr << "fatal: Unrecognized arguments" << std::endl;
    return 4;
  }

  const fs::path path(FLAGS_path);
  const vector<fs::path> files = programl::task::dataflow::EnumerateIrLists(path / "ir");

  programl::task::dataflow::ParallelFileMap<programl::task::dataflow::ProcessIrList, 128>(path,
                                                                                          files);
  LOG(INFO) << "done";

  return 0;
}
