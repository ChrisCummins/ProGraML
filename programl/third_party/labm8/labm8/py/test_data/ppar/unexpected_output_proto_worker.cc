// A program which reads an AddXandY message from stdin, and writes an
// UnexpectedOutputProto message to stdout.
#include <iostream>

#include "labm8/cpp/logging.h"
#include "labm8/py/test_data/ppar/protos.pb.h"

int main() {
  // The proto instance that we'll parse from istream.
  AddXandY input_message;
  // The proto instance that we'll store the result in.
  UnexpectedOutputProto output_message;

  // Decode the proto from istream.
  CHECK(input_message.ParseFromIstream(&std::cin));

  output_message.set_unexpected_message("Well this is unexpected!");

  // Write the message to ostream.
  CHECK(output_message.SerializeToOstream(&std::cout));
}
