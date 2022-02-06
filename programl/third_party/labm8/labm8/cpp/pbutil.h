// Utility code for working with protocol buffers.
#pragma once

#include <fstream>
#include <functional>
#include <iostream>

#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/text_format.h"
#include "labm8/cpp/logging.h"

namespace labm8 {
namespace pbutil {

// Write a protocol message to file in text format.
template <typename Message>
void TextFormatProtoToFile(const string& path, const Message& proto) {
  std::ofstream out(path);
  out << proto.DebugString();
}

// Read a text format protocol message from stdin. If parsing fails, terminate.
template <typename Message>
Message ReadTextFormatProtoFromStdinOrDie() {
  google::protobuf::io::IstreamInputStream istream(&std::cin);
  Message message;
  if (!google::protobuf::TextFormat::Parse(&istream, &message)) {
    std::cerr << "fatal: failed to parse stdin";
    exit(3);
  }
  return message;
}

// Run a process_function callback that accepts a proto message and mutates
// it in place. The proto message is decoded from the given istream, and
// serialized to to the ostream.
template <typename Message>
void ProcessMessageInPlace(std::function<void(Message*)> process_function,
                           std::istream* istream = &std::cin, std::ostream* ostream = &std::cout) {
  // The proto instance that we'll parse from istream.
  Message message;

  // Decode the proto from istream.
  CHECK(message.ParseFromIstream(istream));

  // Do the work.
  process_function(&message);

  // Write the message to ostream.
  CHECK(message.SerializeToOstream(ostream));
}

// Run a process_function callback that accepts a proto message and writes
// to an output proto message. The input proto message is decoded from the given
// istream, and the output proto is serialized to to the ostream.
template <typename InputMessage, typename OutputMessage>
void ProcessMessage(std::function<void(const InputMessage&, OutputMessage*)> process_function,
                    std::istream* istream = &std::cin, std::ostream* ostream = &std::cout) {
  // The proto instance that we'll parse from istream.
  InputMessage input_message;
  // The proto instance that we'll store the result in.
  OutputMessage output_message;

  // Decode the proto from istream.
  CHECK(input_message.ParseFromIstream(istream));

  // Do the work.
  process_function(input_message, &output_message);

  // Write the message to ostream.
  CHECK(output_message.SerializeToOstream(ostream));
}

}  // namespace pbutil
}  // namespace labm8

// A convenience macro to run an in-place process_function as the main()
// function of a program.
#define PBUTIL_INPLACE_PROCESS_MAIN(process_function, message_type)       \
  int main() {                                                            \
    labm8::pbutil::ProcessMessageInPlace<message_type>(process_function); \
    return 0;                                                             \
  }

// A convenience macro to run an process_function as the main() function of a
// program.
#define PBUTIL_PROCESS_MAIN(process_function, input_message_type, output_message_type)        \
  int main() {                                                                                \
    labm8::pbutil::ProcessMessage<input_message_type, output_message_type>(process_function); \
    return 0;                                                                                 \
  }
