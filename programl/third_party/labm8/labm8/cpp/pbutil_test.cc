#include "labm8/cpp/pbutil.h"

#include <sstream>

#include "labm8/cpp/test.h"
#include "labm8/cpp/test/protos.pb.h"

namespace labm8 {
namespace pbutil {

TEST(ProcessMessageInPlace, SetAField) {
  // Prepare the input.
  AddXandY message;
  message.set_x(2);
  message.set_y(2);

  std::stringstream istream;
  std::stringstream ostream;

  // In the process function.
  message.SerializeToOstream(&istream);
  ProcessMessageInPlace<AddXandY>(
      [](AddXandY* message) { message->set_result(message->x() + message->y()); }, &istream,
      &ostream);
  message.ParseFromIstream(&ostream);

  // Check the values produced.
  EXPECT_EQ(message.x(), 2);
  EXPECT_EQ(message.y(), 2);
  EXPECT_EQ(message.result(), 4);
}

TEST(ProcessMessageInPlace, MutateAField) {
  // Prepare the input.
  AddXandY message;
  message.set_x(10);

  std::stringstream istream;
  std::stringstream ostream;

  // In the process function.
  message.SerializeToOstream(&istream);
  ProcessMessageInPlace<AddXandY>([](AddXandY* message) { message->set_x(5); }, &istream, &ostream);
  message.ParseFromIstream(&ostream);

  // Check the values produced.
  EXPECT_EQ(message.x(), 5);
  EXPECT_EQ(message.y(), 0);
  EXPECT_EQ(message.result(), 0);
}

TEST(ProcessMessage, AddXandY) {
  // Prepare the input.
  AddXandY message;
  message.set_x(2);
  message.set_y(2);

  std::stringstream istream;
  std::stringstream ostream;

  // In the process function.
  message.SerializeToOstream(&istream);
  ProcessMessage<AddXandY, AddXandY>(
      [](const AddXandY& input, AddXandY* output) { output->set_result(input.x() + input.y()); },
      &istream, &ostream);
  message.ParseFromIstream(&ostream);

  // Check the values produced.
  EXPECT_EQ(message.x(), 0);
  EXPECT_EQ(message.y(), 0);
  EXPECT_EQ(message.result(), 4);
}

void BM_ProcessMessageInPlace(benchmark::State& state) {
  std::stringstream istream;
  std::stringstream ostream;

  while (state.KeepRunning()) {
    std::stringstream istream;
    std::stringstream ostream;
    ProcessMessageInPlace<AddXandY>([](AddXandY* message) { message->set_x(5); }, &istream,
                                    &ostream);
  }
}
BENCHMARK(BM_ProcessMessageInPlace);

void BM_ProcessMessage(benchmark::State& state) {
  std::stringstream istream;
  std::stringstream ostream;

  while (state.KeepRunning()) {
    std::stringstream istream;
    std::stringstream ostream;
    ProcessMessage<AddXandY, AddXandY>(
        [](const AddXandY& input, AddXandY* output) { output->set_result(input.x() + input.y()); },
        &istream, &ostream);
  }
}
BENCHMARK(BM_ProcessMessage);

}  // namespace pbutil
}  // namespace labm8

TEST_MAIN();
