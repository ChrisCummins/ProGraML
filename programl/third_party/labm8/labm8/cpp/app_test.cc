#include "labm8/cpp/app.h"

#include "labm8/cpp/test.h"

namespace labm8 {
namespace {

TEST(InitApp, DoesNotCrash) {
  int argc = 1;
  char* args[] = {"myapp"};
  char** argv = args;
  InitApp(&argc, &argv, "My usage string");
}

}  // anonymous namespace
}  // namespace labm8

TEST_MAIN();
