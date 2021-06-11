// The Feature protocol buffer is a bit cumbersome to work with.
// This file contains utility functions for creating and manipulating
// features.
//
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

#include <string>
#include <vector>

#include "programl/third_party/tensorflow/features.pb.h"

namespace programl {
namespace graph {

// Create a feature protocol buffer with a scalar int64 value.
Feature CreateFeature(int64_t value);

// Create a feature protocol buffer with a list of int64 values.
Feature CreateFeature(const std::vector<int64_t>& value);

// Create a feature protocol buffer with a scalar string value.
Feature CreateFeature(const std::string& value);

// Set the feature value to the given message.
void SetFeature(Features* features, const char* label, const Feature& value);

// Convenience function to add a scalar feature value.
template <typename MessageType, typename FeatureType>
void AddScalarFeature(MessageType* message, const std::string& key, const FeatureType& value) {
  message->mutable_features()->mutable_feature()->insert({key, CreateFeature(value)});
}

}  // namespace graph
}  // namespace programl
