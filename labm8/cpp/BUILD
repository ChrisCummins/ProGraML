# Core C++ libraries.
#
# Copyright 2014-2019 Chris Cummins <chrisc.101@gmail.com>.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cc_library(
    name = "app",
    srcs = ["app.cc"],
    hdrs = ["app.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_github_gflags_gflags//:gflags",
    ],
)

cc_test(
    name = "app_test",
    srcs = ["app_test.cc"],
    deps = [
        ":app",
        ":test",
    ],
)

cc_library(
    name = "bazelutil",
    srcs = ["bazelutil.cc"],
    hdrs = ["bazelutil.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":logging",
        ":string",
        "@boost//:filesystem",
    ],
)

cc_test(
    name = "bazelutil_test",
    srcs = ["bazelutil_test.cc"],
    data = ["//labm8/cpp/test:data_file"],
    deps = [
        ":bazelutil",
        ":test",
    ],
)

cc_library(
    name = "callback",
    hdrs = ["callback.h"],
    visibility = ["//visibility:public"],
    deps = [":macros"],
)

cc_library(
    name = "casts",
    hdrs = ["casts.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "common",
    srcs = ["common.cc"],
    hdrs = ["common.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":callback",
        ":logging",
        ":macros",
        ":mutex",
        ":once",
        ":port",
        ":status",
    ],
)

cc_test(
    name = "common_test",
    srcs = ["common_test.cc"],
    deps = [
        ":callback",
        ":casts",
        ":int128",
        ":logging",
        ":mutex",
        ":once",
        ":status",
        ":stringpiece",
        ":strutil",
        ":substitute",
        ":test",
    ],
)

cc_library(
    name = "int128",
    srcs = ["int128.cc"],
    hdrs = ["int128.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":common",
        ":logging",
        ":port",
    ],
)

cc_test(
    name = "int128_test",
    srcs = ["int128_test.cc"],
    deps = [
        ":int128",
        ":logging",
        ":test",
    ],
)

cc_library(
    name = "logging",
    srcs = ["logging.cc"],
    hdrs = ["logging.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":macros",
        ":port",
        ":string",
        ":stringpiece",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "macros",
    hdrs = ["macros.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":port",
    ],
)

cc_library(
    name = "mutex",
    hdrs = ["mutex.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "once",
    hdrs = ["once.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "pbutil",
    hdrs = ["pbutil.h"],
    visibility = ["//visibility:public"],
    deps = [":logging"],
)

cc_test(
    name = "pbutil_test",
    srcs = ["pbutil_test.cc"],
    deps = [
        ":pbutil",
        ":test",
        "//labm8/cpp/test:protos_pb_cc",
    ],
)

cc_library(
    name = "platform_macros",
    hdrs = ["platform_macros.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "port",
    hdrs = ["port.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":platform_macros",
    ],
)

cc_library(
    name = "status",
    srcs = ["status.cc"],
    hdrs = ["status.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":string",
        ":stringpiece",
    ],
)

cc_test(
    name = "status_test",
    srcs = ["status_test.cc"],
    deps = [
        ":logging",
        ":status",
        ":test",
    ],
)

cc_library(
    name = "status_macros",
    hdrs = ["status_macros.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":common",
        ":status",
        ":statusor",
    ],
)

cc_library(
    name = "statusor",
    srcs = ["statusor.cc"],
    hdrs = ["statusor.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":logging",
        ":status",
    ],
)

cc_test(
    name = "statusor_test",
    srcs = ["statusor_test.cc"],
    deps = [
        ":statusor",
        "//labm8/cpp:test",
    ],
)

cc_library(
    name = "stl_util",
    hdrs = ["stl_util.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "string",
    srcs = ["string.cc"],
    hdrs = ["string.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@boost//:algorithm",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "string_test",
    srcs = ["string_test.cc"],
    deps = [
        ":string",
        ":test",
    ],
)

cc_library(
    name = "stringpiece",
    srcs = ["stringpiece.cc"],
    hdrs = ["stringpiece.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":string",
    ],
)

cc_test(
    name = "stringpiece_test",
    srcs = ["stringpiece_test.cc"],
    deps = [
        ":common",
        ":logging",
        ":stringpiece",
        ":test",
    ],
)

cc_library(
    name = "strtod",
    srcs = ["strtod.cc"],
    hdrs = ["strtod.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":common",
        ":logging",
    ],
)

cc_library(
    name = "strutil",
    srcs = ["strutil.cc"],
    hdrs = ["strutil.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":logging",
        ":stl_util",
        ":stringpiece",
        ":strtod",
    ],
)

cc_test(
    name = "strutil_test",
    srcs = ["strutil_test.cc"],
    deps = [
        ":stl_util",
        ":strutil",
        ":test",
    ],
)

cc_library(
    name = "substitute",
    srcs = ["substitute.cc"],
    hdrs = ["substitute.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":logging",
        ":stl_util",
        ":string",
        ":strutil",
    ],
)

cc_library(
    name = "test",
    hdrs = ["test.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":app",
        "@com_github_gflags_gflags//:gflags",
        "@com_github_google_benchmark//:benchmark",
        "@gtest",
    ],
)
