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
#include "programl/util/nproc.h"

#include <string>

#include "subprocess/subprocess.hpp"

namespace programl {
namespace util {

size_t GetNumberOfProcessors() {
  auto out = subprocess::check_output("nproc");
  return std::stoi(std::string(out.buf.begin(), out.buf.end()));
}

}  // namespace util
}  // namespace programl
