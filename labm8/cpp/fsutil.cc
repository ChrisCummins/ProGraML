#include "labm8/cpp/fsutil.h"

#include <iostream>

#include "labm8/cpp/logging.h"

namespace labm8 {
namespace fsutil {

fs::path GetHomeDirectoryOrDie() {
  const char* home = getenv("HOME");
  CHECK(home) << "$HOME not set";
  fs::path path(home);
  CHECK(fs::is_directory(path)) << "Directory not found: " << home;
  return path;
}

Status ReadFile(const fs::path& path, string* contents) {
  std::ifstream ifs(path.string().c_str());
  ifs.seekg(0, std::ios::end);
  contents->resize(ifs.tellg());
  ifs.seekg(0);
  ifs.read(&(*contents)[0], contents->size());
  return Status::OK;
}

Status EnumerateFilesRecursively(const fs::path& root,
                                 vector<fs::path>* files) {
  if (!fs::is_directory(root)) {
    return Status(error::Code::INVALID_ARGUMENT, "Directory not found: {}",
                  root.string());
  }

  for (auto it : fs::recursive_directory_iterator(root)) {
    if (!fs::is_regular_file(it)) {
      continue;
    }
    files->push_back(it.path());
  }
  return Status::OK;
}

}  // namespace fsutil
}  // namespace labm8
