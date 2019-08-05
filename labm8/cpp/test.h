// Header file for unit testing.
//
// All C++ unit tests files should include this header, which will pull in
// gtest and benchmark libraries.
#pragma once

#include "labm8/cpp/app.h"

#include "benchmark/benchmark.h"
#include "gflags/gflags.h"
#include "gtest/gtest.h"

#ifdef TEST_MAIN
#error "TEST_MAIN already defined!"
#endif

DEFINE_string(test_coverage_data_dir, "",
              "TODO(cec): An unused flag added to provide usage compatbility "
              "with python tests that use this flag.");

// Inserts a main() function which runs google benchmarks and gtest suite.
#define TEST_MAIN()                                      \
  int main(int argc, char **argv) {                      \
    testing::InitGoogleTest(&argc, argv);                \
    labm8::InitApp(&argc, &argv, "Test suite program."); \
    const auto ret = RUN_ALL_TESTS();                    \
    benchmark::Initialize(&argc, argv);                  \
    benchmark::RunSpecifiedBenchmarks();                 \
    return ret;                                          \
  }
