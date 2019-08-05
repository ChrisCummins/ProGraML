#include "labm8/cpp/app.h"

namespace labm8 {

void InitApp(int* argc, char*** argv, const char* usage_string) {
  std::string usage(usage_string);
  if (!usage.empty()) {
    gflags::SetUsageMessage(usage);
  }

  gflags::ParseCommandLineFlags(argc, argv, /*remove_flags=*/true);
}

}  // namespace labm8
