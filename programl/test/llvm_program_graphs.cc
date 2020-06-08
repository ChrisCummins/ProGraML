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
#include "programl/test/llvm_program_graphs.h"

namespace programl {
namespace test {

namespace fs = boost::filesystem;

template <typename ProtocolBuffer>
vector<ProtocolBuffer> ReadDirectoryOfProtos(const fs::path& path) {
  vector<ProtocolBuffer> protos;

  for (auto it : fs::directory_iterator(path)) {
    if (!labm8::HasSuffixString(it.path().string(), ".pb")) {
      continue;
    }
    std::ifstream file(it.path().string());
    ProtocolBuffer message;
    CHECK(message.ParseFromIstream(&file));
    protos.push_back(message);
  }

  LOG(INFO) << "Read " << protos.size() << " files";

  return protos;
}

vector<ProgramGraph> ReadLlvmProgramGraphs() {
  const auto path =
      labm8::BazelDataPathOrDie("phd/programl/test/data/llvm_ir_graphs");
  return ReadDirectoryOfProtos<ProgramGraph>(path);
}

}  // namespace test
}  // namespace programl
