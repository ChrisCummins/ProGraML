# A package for the LLVM headers.

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "headers",
    hdrs = glob([
        # LLVM uses some funky file extensions.
        "**/*.h",
        "**/*.inc",
        "**/*.def",
        "**/*.gen",
    ]),
)
