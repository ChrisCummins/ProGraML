#include "labm8/cpp/bazelutil.h"

#include "labm8/cpp/logging.h"

namespace labm8 {

namespace {

// Return absolute canonocial representation of an input path.
boost::filesystem::path CanonicalPath(const boost::filesystem::path& path) {
  return boost::filesystem::canonical(boost::filesystem::absolute(path));
}

}  // anonymous namespace

boost::filesystem::path BazelDataPathOrDie(const string& path) {
  boost::filesystem::path fs_path("../" + path);
  CHECK(boost::filesystem::is_regular_file(fs_path))
      << "Bazel data path '" << CanonicalPath(fs_path).c_str() << "' not found";
  return CanonicalPath(fs_path);
}

}  // namespace labm8
