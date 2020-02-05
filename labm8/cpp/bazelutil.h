#pragma once

#include "labm8/cpp/statusor.h"
#include "labm8/cpp/string.h"

#include "boost/filesystem.hpp"

namespace labm8 {

// Return the absolute path to a data file.
//
// This provides access to files from the 'data' attribute of a target in
// Bazel. Given a fully relative path to a data file,
// e.g. "phd/my/package/data", return the absolute path. The path must be
// relative to the bazel runfiles root, and begin with the name of the
// workspace.
//
StatusOr<boost::filesystem::path> BazelDataPath(const string& path);
boost::filesystem::path BazelDataPathOrDie(const string& path);

}  // namespace labm8
