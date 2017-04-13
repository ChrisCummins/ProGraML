workspace(name="phd")

new_http_archive(
    name = "gtest",
    url = "https://github.com/google/googletest/archive/release-1.8.0.zip",
    sha256 = "f3ed3b58511efd272eb074a3a6d6fb79d7c2e6a0e374323d1e6bcbcc1ef141bf",
    build_file = "gtest.BUILD",
    strip_prefix = "googletest-release-1.8.0/googletest",
)

new_http_archive(
    name = "benchmark",
    url = "https://github.com/google/benchmark/archive/v1.1.0.tar.gz",
    sha256 = "e7334dd254434c6668e33a54c8f839194c7c61840d52f4b6258eee28e9f3b20e",
    build_file = "benchmark.BUILD",
    strip_prefix = "benchmark-1.1.0",
)

# FIXME: Mac hard-coded
new_http_archive(
    name = "tbb",
    url = "https://github.com/01org/tbb/releases/download/2017_U5/tbb2017_20170226oss_mac.tgz",
    sha256 = "6ff553ec31c33b8340ce2113853be1c42e12b1a4571f711c529f8d4fa762a1bf",
    build_file = "tbb.BUILD",
    strip_prefix = 'tbb2017_20170226oss',
)
