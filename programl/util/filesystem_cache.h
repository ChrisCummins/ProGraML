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
#pragma once

#include <cstdlib>
#include <vector>

#include "boost/filesystem.hpp"
#include "labm8/cpp/logging.h"

namespace programl {
namespace util {

// A filesystem cache. This is hardcoded to ~/.cache/programl.
class FilesystemCache {
 public:
  FilesystemCache();

  // Convert the given key into a local filesystem path.
  boost::filesystem::path operator[](const std::vector<std::string>& components) const;

 private:
  const boost::filesystem::path root_;
};

}  // namespace util
}  // namespace programl
