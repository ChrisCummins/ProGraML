load("@io_bazel_rules_docker//python3:image.bzl", "py3_image")

py_test(
    name = "basic_app_test",
    srcs = ["basic_app_test.py"],
    deps = [
        "//labm8:app",
    ],
)

py3_image(
    name = "basic_app_test_image",
    testonly = 1,
    srcs = ["basic_app_test.py"],
    base = "@base//image",
    main = "basic_app_test.py",
    deps = [":basic_app_test"],
)

sh_test(
    name = "basic_app_docker_test",
    srcs = ["basic_app_test.sh"],
    data = [
        ":basic_app_test_image.tar",
    ],
)

py_test(
    name = "mysql_test",
    srcs = ["mysql_test.py"],
    deps = [
        "//labm8:test",
        "//third_party/py/mysql",
    ],
)

py3_image(
    name = "mysql_test_image",
    testonly = 1,
    srcs = ["mysql_test.py"],
    base = "@base//image",
    main = "mysql_test.py",
    deps = [":mysql_test"],
)

sh_test(
    name = "mysql_docker_test",
    srcs = ["mysql_test.sh"],
    data = [
        ":mysql_test_image.tar",
    ],
)
