config_setting(
    name = "darwin",
    values = {"cpu": "darwin"},
    visibility = ["//visibility:public"],
)

filegroup(
    name = "config",
    srcs = ["config.pbtxt"],
    visibility = ["//config:__subpackages__"],
)

filegroup(
    name = "config_script",
    srcs = ["configure"],
    visibility = ["//visibility:public"],
)
