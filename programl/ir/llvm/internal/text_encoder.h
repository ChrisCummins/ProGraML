// This file defines utilities for encoding LLVM objects to text.
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

#include <utility>

#include "absl/container/flat_hash_map.h"
#include "labm8/cpp/string.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"

namespace programl {
namespace ir {
namespace llvm {
namespace internal {

struct LlvmTextComponents {
  // The entire text of the instruction, e.g. "%5 = add nsw i32 %3, %4"
  string text;
  // The left-hand side of the instruction, e.g. "i32* %5". If there is no
  // left-hand side, this string is empty.
  string lhs;
  // The right-hand side of the instruction, e.g. "add nsw i32 %3, %4"
  string rhs;
  // The identifier of the left-hand side of the instruction, e.g. "%5". If
  // there is no left-hand side, this string is empty.
  string lhs_identifier;
  // The type of the left-hand side of the instruction, e.g. "i32*". If there is
  // no left-hand side, this string is empty.
  string lhs_type;
  // The type of the right-hand side of the instruction, e.g. "add".
  string opcode_name;
};

class TextEncoder {
 public:
  // Encode the string components describing an LLVM IR object.
  //
  // LLVM doesn't require "names" for instructions since it is in SSA form, so
  // these methods generates one by printing the instruction to a string (to
  // generate identifiers), then splitting the LHS identifier name and
  // concatenating it with the type.
  //
  // See: https://lists.llvm.org/pipermail/llvm-dev/2010-April/030726.html
  LlvmTextComponents Encode(const ::llvm::Instruction* instruction);
  // Fields set: text, lhs, and lhs_type.
  LlvmTextComponents Encode(const ::llvm::Constant* constant);
  // Fields set: text, lhs, and lhs_type.
  LlvmTextComponents Encode(const ::llvm::Argument* argument);
  LlvmTextComponents Encode(const ::llvm::Value* value);
  LlvmTextComponents Encode(const ::llvm::Type* type);

  // Clear the encoded string cache.
  void Clear();

 private:
  // Caches to map LLVM IR objects to their encoded representations. We cache
  // objects since serializing and string manipulation can be expensive, and may
  // need to be performed many times for each object, dependning on its usage.
  absl::flat_hash_map<const ::llvm::Instruction*, LlvmTextComponents> instruction_cache_;
  absl::flat_hash_map<const ::llvm::Constant*, LlvmTextComponents> constant_cache_;
  absl::flat_hash_map<const ::llvm::Argument*, LlvmTextComponents> arg_cache_;
  absl::flat_hash_map<const ::llvm::Value*, LlvmTextComponents> value_cache_;
  absl::flat_hash_map<const ::llvm::Type*, LlvmTextComponents> type_cache_;
};

}  // namespace internal
}  // namespace llvm
}  // namespace ir
}  // namespace programl
