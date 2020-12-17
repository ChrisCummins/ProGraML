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
#include "programl/ir/xla/xla_stringifier.h"

#include <sstream>

namespace programl {
namespace ir {
namespace xla {

string ShapeProtoToString(const ::xla::ShapeProto& shape) {
  std::stringstream str;
  str << PrimitiveType_Name(shape.element_type());
  for (int i = 0; i < shape.dimensions_size(); ++i) {
    str << (i ? 'x' : ' ');
    str << shape.dimensions(i);
  }
  return str.str();
}

string HloInstructionToText(const ::xla::HloInstructionProto& instruction) {
  if (instruction.opcode() == "parameter" || instruction.opcode() == "constant") {
    return instruction.name();
  }

  return instruction.opcode();
}

string LiteralProtoToText(const ::xla::LiteralProto& literal) {
  return ShapeProtoToString(literal.shape());
}

}  // namespace xla
}  // namespace ir
}  // namespace programl
