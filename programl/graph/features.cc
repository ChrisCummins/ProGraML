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

#include "programl/graph/features.h"

namespace programl {
namespace graph {

Feature CreateFeature(int64_t value) {
  Feature feature;
  feature.mutable_int64_list()->add_value(value);
  return feature;
}

Feature CreateFeature(const std::vector<int64_t>& value) {
  Feature feature;
  for (auto v : value) {
    feature.mutable_int64_list()->add_value(v);
  }
  return feature;
}

Feature CreateFeature(const std::string& value) {
  Feature feature;
  feature.mutable_bytes_list()->add_value(value);
  return feature;
}

void SetFeature(Features* features, const char* label, const Feature& value) {
  (*features->mutable_feature())[label] = value;
}

}  // namespace graph
}  // namespace programl
