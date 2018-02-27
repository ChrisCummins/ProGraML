load("@requirements//:requirements.bzl", "requirement")

py_library(
    name = "labm8",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "cache",
    srcs = ["cache.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":crypto",
        ":fs",
        ":io",
        requirement("six"),
    ],
)

py_library(
    name = "crypto",
    srcs = ["crypto.py"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "db",
    srcs = ["db.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":fs",
        ":io",
        requirement("six"),
    ],
)

py_library(
    name = "dirhashcache",
    srcs = ["dirhashcache.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":fs",
        requirement("checksumdir"),
    ],
)

py_library(
    name = "fs",
    srcs = ["fs.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":labm8",
        ":labtypes",
        requirement("humanize"),
        requirement("Send2Trash"),
    ],
)

py_library(
    name = "io",
    srcs = ["io.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":labtypes",
        requirement("humanize"),
        requirement("Send2Trash"),
    ],
)

py_library(
    name = "jsonutil",
    srcs = ["jsonutil.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":fs",
    ],
)

py_library(
    name = "latex",
    srcs = ["latex.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":io",
    ],
)

py_library(
    name = "lockfile",
    srcs = ["lockfile.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":fs",
        ":system",
    ],
)

py_library(
    name = "make",
    srcs = ["make.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":fs",
        ":system",
    ],
)

py_library(
    name = "labmath",
    srcs = ["labmath.py"],
    visibility = ["//visibility:public"],
    deps = [
        requirement("numpy"),
        requirement("scipy"),
    ],
)

py_library(
    name = "modules",
    srcs = ["modules.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":io",
    ],
)

py_library(
    name = "prof",
    srcs = ["prof.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":labtypes",
    ],
)

py_library(
    name = "system",
    srcs = ["system.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":fs",
    ],
)

py_library(
    name = "tar",
    srcs = ["tar.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":fs",
    ],
)

py_library(
    name = "text",
    srcs = ["text.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":fs",
        ":system",
    ],
)

py_library(
    name = "time",
    srcs = ["time.py"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "labtypes",
    srcs = ["labtypes.py"],
    visibility = ["//visibility:public"],
    deps = [requirement("six")],
)

py_library(
    name = "viz",
    srcs = ["viz.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":io",
    ],
)
