# LLVM package.

load("@com_github_chriscummins_rules_bats//:bats.bzl", "bats_test")

sh_binary(
    name = "clang",
    srcs = ["clang.sh"],
    data = select({
        "//:darwin": [
            "@llvm_mac//:clang",
        ],
        "//conditions:default": [
            "@llvm_linux//:clang",
        ],
    }),
    visibility = ["//visibility:public"],
    deps = [
        "//labm8/sh:app",
    ],
)

bats_test(
    name = "clang_smoke_test",
    srcs = ["clang_smoke_test.bats"],
    data = [
        ":clang",
        "//labm8/sh:test",
    ],
)

sh_binary(
    name = "clang-format",
    srcs = ["clang-format.sh"],
    data = select({
        "//:darwin": [
            "@llvm_mac//:clang-format",
        ],
        "//conditions:default": [
            "@llvm_linux//:clang-format",
        ],
    }),
    visibility = ["//visibility:public"],
    deps = [
        "//labm8/sh:app",
    ],
)

bats_test(
    name = "clang-format_smoke_test",
    srcs = ["clang-format_smoke_test.bats"],
    data = [
        ":clang-format",
        "//labm8/sh:test",
    ],
)

sh_binary(
    name = "llvm-as",
    srcs = ["llvm-as.sh"],
    data = select({
        "//:darwin": [
            "@llvm_mac//:llvm-as",
        ],
        "//conditions:default": [
            "@llvm_linux//:llvm-as",
        ],
    }),
    visibility = ["//visibility:public"],
    deps = [
        "//labm8/sh:app",
    ],
)

bats_test(
    name = "llvm-as_smoke_test",
    srcs = ["llvm-as_smoke_test.bats"],
    data = [
        ":llvm-as",
        "//labm8/sh:test",
    ],
)

sh_binary(
    name = "llvm-dis",
    srcs = ["llvm-dis.sh"],
    data = select({
        "//:darwin": [
            "@llvm_mac//:llvm-dis",
        ],
        "//conditions:default": [
            "@llvm_linux//:llvm-dis",
        ],
    }),
    visibility = ["//visibility:public"],
    deps = [
        "//labm8/sh:app",
    ],
)

bats_test(
    name = "llvm-dis_smoke_test",
    srcs = ["llvm-dis_smoke_test.bats"],
    data = [
        ":llvm-dis",
        "//labm8/sh:test",
    ],
)

sh_binary(
    name = "llvm-link",
    srcs = ["llvm-link.sh"],
    data = select({
        "//:darwin": [
            "@llvm_mac//:llvm-link",
        ],
        "//conditions:default": [
            "@llvm_linux//:llvm-link",
        ],
    }),
    visibility = ["//visibility:public"],
    deps = [
        "//labm8/sh:app",
    ],
)

bats_test(
    name = "llvm-link_smoke_test",
    srcs = ["llvm-link_smoke_test.bats"],
    data = [
        ":llvm-link",
        "//labm8/sh:test",
    ],
)

sh_binary(
    name = "opt",
    srcs = ["opt.sh"],
    data = select({
        "//:darwin": [
            "@llvm_mac//:opt",
        ],
        "//conditions:default": [
            "@llvm_linux//:opt",
        ],
    }),
    visibility = ["//visibility:public"],
    deps = [
        "//labm8/sh:app",
    ],
)

bats_test(
    name = "opt_smoke_test",
    srcs = ["opt_smoke_test.bats"],
    data = [
        ":opt",
        "//labm8/sh:test",
    ],
)
