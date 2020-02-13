// Header file for unit testing.
//
// All C++ unit tests files should include this header, which will pull in
// gtest and benchmark libraries.
#pragma once

#include <fstream>
#include <string>

#include "boost/filesystem.hpp"

#include "labm8/cpp/app.h"

#include "benchmark/benchmark.h"
#include "gflags/gflags.h"
#include "gtest/gtest.h"

#ifdef TEST_MAIN
#error "TEST_MAIN already defined!"
#endif

namespace labm8 {

// Get the directory of the directory for temporary files. If TEST_TMPDIR
// environment variable is set, it is used. Else, boost finds a suitable
// temporary filesystem location (e.g. /tmp).
boost::filesystem::path GetTestTempdir();

// Base class for implementing tests.
class Test : public ::testing::Test {
 protected:
  Test() : tempDir_(GetTestTempdir()) {}

  // Return the path of the temporary directory.
  boost::filesystem::path GetTempDir() const;

  // Return the absolute path of a file within the temporary directory with a
  // random name. This does not create the file.
  boost::filesystem::path GetTempFile(
      const std::string& suffix = ".tmpfile") const;

 private:
  const boost::filesystem::path tempDir_;
};

}  // namespace labm8

// Inserts a main() function which runs google benchmarks and gtest suite.
#define TEST_MAIN()                                      \
  int main(int argc, char** argv) {                      \
    testing::InitGoogleTest(&argc, argv);                \
    labm8::InitApp(&argc, &argv, "Test suite program."); \
    const auto ret = RUN_ALL_TESTS();                    \
    benchmark::Initialize(&argc, argv);                  \
    benchmark::RunSpecifiedBenchmarks();                 \
    return ret;                                          \
  }
