// A binary that reads an AddXandY message from stdin, and writes an AddXandY
// message to stdout.
//
// If the input AddXandY.x == 10, the program crashes.

#include "labm8/cpp/logging.h"
#include "labm8/cpp/pbutil.h"
#include "labm8/py/test_data/ppar/protos.pb.h"

void ProcessProtobuf(const AddXandY& input_proto, AddXandY* output_proto) {
  int x = input_proto.x();
  int y = input_proto.y();

  CHECK(x != 10);

  LOG(INFO) << "Adding " << x << " and " << y << " and storing the result in a new message";
  output_proto->set_result(x + y);
}

PBUTIL_PROCESS_MAIN(ProcessProtobuf, AddXandY, AddXandY);
