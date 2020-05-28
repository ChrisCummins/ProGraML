#include "labm8/cpp/bazelutil.h"

#include "labm8/cpp/logging.h"
#include "labm8/cpp/status.h"

#include <sstream>

namespace labm8 {

namespace {

// Return absolute canonocial representation of an input path.
boost::filesystem::path CanonicalPath(const boost::filesystem::path& path) {
  return boost::filesystem::canonical(boost::filesystem::absolute(path));
}

}  // anonymous namespace

StatusOr<boost::filesystem::path> BazelDataPath(const string& path) {
  boost::filesystem::path fs_path("../" + path);
  if (!boost::filesystem::exists(fs_path)) {
    return labm8::Status(labm8::error::Code::INVALID_ARGUMENT,
                         "Bazel data path '{}' not found",
                         CanonicalPath(fs_path).c_str());
  }
  return CanonicalPath(fs_path);
}

boost::filesystem::path BazelDataPathOrDie(const string& path) {
  return BazelDataPath(path).ValueOrDie();
}

}  // namespace labm8
