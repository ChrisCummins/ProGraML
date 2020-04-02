#include "labm8/cpp/crypto.h"

#include <stdio.h>
#include <array>
#include <iostream>
#include <memory>
#include "boost/filesystem.hpp"
#include "labm8/cpp/status.h"
#include "subprocess/subprocess.hpp"

namespace error = labm8::error;
namespace fs = boost::filesystem;

namespace labm8 {
namespace crypto {

namespace {

StatusOr<string> Exec(const string &cmd, size_t expectedLength) {
  auto process = subprocess::Popen(cmd, subprocess::shell{true},
                                   subprocess::output{subprocess::PIPE},
                                   subprocess::error{subprocess::PIPE});
  auto stdout = process.communicate().first;
  if (process.retcode()) {
    return Status(error::Code::FAILED_PRECONDITION, "failed: {}", cmd);
  }
  string checksum = string(stdout.buf.begin(), stdout.buf.end());
  if (checksum.size() != expectedLength) {
    return Status(error::Code::FAILED_PRECONDITION, "invalid return length");
  }
  checksum.erase(checksum.size() - 1);
  return checksum;
}

StatusOr<string> Exec(const string &cmd, size_t expectedLength,
                      const string &input) {
  auto process = subprocess::Popen(cmd, subprocess::shell{true},
                                   subprocess::input{subprocess::PIPE},
                                   subprocess::output{subprocess::PIPE},
                                   subprocess::error{subprocess::PIPE});
  auto stdout = process.communicate(input.c_str(), input.size()).first;
  if (process.retcode()) {
    return Status(error::Code::FAILED_PRECONDITION, "failed: {}", cmd);
  }
  string checksum = string(stdout.buf.begin(), stdout.buf.end());
  if (checksum.size() != expectedLength) {
    return Status(error::Code::FAILED_PRECONDITION, "invalid return length");
  }
  checksum.erase(checksum.size() - 1);
  return checksum;
}

}  // anonymous namespace

StatusOr<string> Sha1(const fs::path &path) {
  return Exec("sha1sum " + path.string() + " | cut -f1 -d' '", 41);
}

StatusOr<string> Sha256(const fs::path &path) {
  return Exec("sha256sum " + path.string() + " | cut -f1 -d' '", 65);
}

StatusOr<string> Sha256(const string &text) {
  return Exec("sha256sum | cut -f1 -d' '", 65, text);
}

}  // namespace crypto
}  // namespace labm8
