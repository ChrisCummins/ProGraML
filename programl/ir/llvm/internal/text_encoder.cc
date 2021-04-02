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
#include "programl/ir/llvm/internal/text_encoder.h"

#include <sstream>
#include <utility>

#include "labm8/cpp/logging.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/raw_ostream.h"

namespace programl {
namespace ir {
namespace llvm {
namespace internal {

namespace {

// Produce the textual representation of an LLVM object.
// Generic implementation works for values, instructions, etc.
template <typename T>
string PrintToString(const T& value) {
  string str;
  ::llvm::raw_string_ostream rso(str);
  value.print(rso);
  // Trim any leading indentation whitespace.
  labm8::TrimLeft(str);
  return str;
}

// Resolve the base type after dereferencing pointer indirection(s).
// If the type is not a pointer, the type argument is returned.
// Argument pointerDepth is incremented for every level of pointer
// dereferencing.
const ::llvm::Type* GetDereferencedType(const ::llvm::Type* type, int* pointerDepth) {
  CHECK(type) << "nullptr type at pointer depth " << *pointerDepth;
  if (type->isPointerTy()) {
    *pointerDepth = *pointerDepth + 1;
    return GetDereferencedType(type->getPointerElementType(), pointerDepth);
  } else {
    return type;
  }
}

// Specialization for LLVM types which returns "struct" or "struct*" for
// struct or pointer to struct types, respectively. All other types are
// serialized as normal.
template <>
string PrintToString(const ::llvm::Type& value) {
  string str;

  int pointerDepth = 0;
  if (GetDereferencedType(&value, &pointerDepth)->isStructTy()) {
    str = "struct";
    for (int i = 0; i < pointerDepth; ++i) {
      str.push_back('*');
    }
  } else {
    ::llvm::raw_string_ostream rso(str);
    value.print(rso);
    // Trim any leading indentation whitespace.
    labm8::TrimLeft(str);
  }

  return str;
}

template <typename T>
LlvmTextComponents EncodeAndCache(const T* value,
                                  absl::flat_hash_map<const T*, LlvmTextComponents>* cache) {
  auto it = cache->find(value);
  if (it != cache->end()) {
    return it->second;
  }

  LlvmTextComponents encoded;
  encoded.text = PrintToString(*value);

  cache->insert({value, encoded});
  return encoded;
}

}  // anonymous namespace

LlvmTextComponents TextEncoder::Encode(const ::llvm::Instruction* instruction) {
  // Return from cache if available.
  auto it = instruction_cache_.find(instruction);
  if (it != instruction_cache_.end()) {
    return it->second;
  }

  LlvmTextComponents encoded;
  encoded.text = PrintToString(*instruction);
  encoded.opcode_name = instruction->getOpcodeName(instruction->getOpcode());

  const size_t snipAt = encoded.text.find(" = ");

  // An instruction without a LHS.
  if (snipAt == string::npos) {
    instruction_cache_.insert({instruction, encoded});
    return encoded;
  }

  encoded.lhs_type = PrintToString(*instruction->getType());
  encoded.lhs_identifier = encoded.text.substr(0, snipAt);

  std::stringstream instructionName;
  instructionName << encoded.lhs_type << ' ' << encoded.lhs_identifier;
  encoded.lhs = instructionName.str();

  encoded.rhs = encoded.text.substr(snipAt + 3);

  instruction_cache_.insert({instruction, encoded});
  return encoded;
}

LlvmTextComponents TextEncoder::Encode(const ::llvm::Constant* constant) {
  // Return from cache if available.
  auto it = constant_cache_.find(constant);
  if (it != constant_cache_.end()) {
    return it->second;
  }

  LlvmTextComponents encoded;
  encoded.text = PrintToString(*constant);
  encoded.lhs = encoded.text;
  encoded.lhs_type = PrintToString(*constant->getType());

  constant_cache_.insert({constant, encoded});
  return encoded;
}

LlvmTextComponents TextEncoder::Encode(const ::llvm::Value* value) {
  return EncodeAndCache(value, &value_cache_);
}

LlvmTextComponents TextEncoder::Encode(const ::llvm::Type* type) {
  return EncodeAndCache(type, &type_cache_);
}

LlvmTextComponents TextEncoder::Encode(const ::llvm::Argument* argument) {
  // Return from cache if available.
  auto it = arg_cache_.find(argument);
  if (it != arg_cache_.end()) {
    return it->second;
  }

  LlvmTextComponents encoded;
  encoded.text = PrintToString(*argument);

  encoded.lhs = encoded.text;
  encoded.lhs_type = PrintToString(*argument->getType());

  arg_cache_.insert({argument, encoded});
  return encoded;
}

void TextEncoder::Clear() {
  constant_cache_.clear();
  instruction_cache_.clear();
  arg_cache_.clear();
  value_cache_.clear();
  type_cache_.clear();
}

}  // namespace internal
}  // namespace llvm
}  // namespace ir
}  // namespace programl
