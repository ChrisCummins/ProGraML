# A package for the Linux source tree.

package(default_visibility = ["//visibility:public"])

# Linux sources as filegroups.

filegroup(
    name = "srcs",
    srcs = glob(["**/*.c"]),
)

filegroup(
    name = "hdrs",
    srcs = glob(["**/*.h"]),
)

filegroup(
    name = "includes",
    srcs = glob([
        "include/**/*",
        "arch/x86/include/**/*",
        "arch/ia64/include/**/*",
        "tools/include/**/*",
    ]),
)
