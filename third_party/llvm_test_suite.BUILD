# The LLVM test suite.
# See: https://llvm.org/docs/TestingGuide.html#test-suite-overview

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "queens_srcs",
    srcs = ["SingleSource/Benchmarks/McGill/queens.c"],
)

cc_binary(
    name = "queens",
    srcs = [":queens_srcs"],
)
