#pragma once

#include "boost/filesystem.hpp"
#include "labm8/cpp/statusor.h"
#include "labm8/cpp/string.h"

namespace fs = boost::filesystem;

namespace labm8 {
namespace crypto {

StatusOr<string> Sha1(const fs::path &path);
StatusOr<string> Sha256(const fs::path &path);
StatusOr<string> Sha256(const string &text);

}  // namespace crypto
}  // namespace labm8
