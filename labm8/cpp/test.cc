#include "labm8/cpp/test.h"
#include "boost/filesystem.hpp"

#include <cstdlib>
#include <cstring>
#include <sstream>

DEFINE_string(test_coverage_data_dir, "",
              "TODO(cec): An unused flag added to provide usage compatbility "
              "with python tests that use this flag.");

namespace labm8 {

boost::filesystem::path GetTestTempdir() {
  const char* test_tmpdir = std::getenv("TEST_TMPDIR");

  if (test_tmpdir && test_tmpdir[0] != '\0') {
    return boost::filesystem::path(test_tmpdir);
  }

  return boost::filesystem::temp_directory_path();
}

boost::filesystem::path Test::GetTempDir() const { return tempDir_; }

boost::filesystem::path Test::GetTempFile(const std::string& suffix) const {
  std::stringstream name;
  name << "%%%%_%%%%_%%%%_%%%%" << suffix;
  return GetTempDir() / boost::filesystem::unique_path(name.str());
}

}  // namespace labm8
