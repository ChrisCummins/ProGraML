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
#include "programl/util/filesystem_cache.h"

#include "labm8/cpp/fsutil.h"

using std::string;
using std::vector;
namespace fs = boost::filesystem;

namespace programl {
namespace util {

FilesystemCache::FilesystemCache()
    : root_(labm8::fsutil::GetHomeDirectoryOrDie() / ".cache" / "programl") {
  fs::create_directories(root_);
};

fs::path FilesystemCache::operator[](const vector<string>& components) const {
  fs::path path(root_);
  CHECK(components.size());
  for (auto& component : components) {
    path /= component;
  }
  return path;
}

}  // namespace util
}  // namespace programl
