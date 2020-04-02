// Utility code for interacting with filesystems.
#pragma once

#include <vector>

#include "boost/filesystem.hpp"
#include "labm8/cpp/status.h"
#include "labm8/cpp/string.h"

using std::vector;
namespace fs = boost::filesystem;

namespace labm8 {
namespace fsutil {

// Return the path of the $HOME variable. If the variable is unset, or if the
// value does not point to a directory, exit.
fs::path GetHomeDirectoryOrDie();

// Read the entire contents of a file to the contents string.
[[nodiscard]] Status ReadFile(const fs::path& path, string* contents);

// Recursively enumerate the files in the root directory, recording the results
// to files.
[[nodiscard]] Status EnumerateFilesRecursively(const fs::path& root,
                                               vector<fs::path>* files);

}  // namespace fsutil
}  // namespace labm8