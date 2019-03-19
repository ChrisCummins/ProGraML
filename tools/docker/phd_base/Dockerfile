# Create a minimal Ubuntu environment for bazel-built binaries.
FROM python:3.6-slim
MAINTAINER Chris Cummins <chrisc.101@gmail.com>

# Put Python into /bin, needed by bazel py_binary script entrypoints.
RUN ln -s /usr/local/bin/python3.6 /usr/bin/python

# Install a few necessary bits and pieces:
#   * a standard compiler toolchain which is needed for cldrive harnesses
#    (because the current clang compilation is not yet entirely hemetic and
#     relies on system linker and C headers)
#   * libreadline6-dev to provide libreadline.so which is needed by oclgrind.
#   * lbzip2 which is needed by tar to unpack .tar.bz2 files, which clgen
#     uses to encode corpus archives.
#   * default-libmysqlclient-dev which is needed by
#     //third_party/py/sqlalchemy.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    binutils libgcc-6-dev libc6-dev libreadline6-dev lbzip2 \
    default-libmysqlclient-dev

# Dirty hack to workaround the fact that oclgrind demands libreadline.so.6, but
# the current Ubuntu package provides libreadline.so.7.
RUN ln -s /lib/x86_64-linux-gnu/libreadline.so.7 \
    /lib/x86_64-linux-gnu/libreadline.so.6

# Remove unneeded files to reduce the image size.
RUN rm -rf /usr/share/doc

# Create a non-root user to execute commands with. This is so that generated
# files in mapped volumes will be created as this user, rather than as root.
RUN useradd -ms /bin/bash docker
USER docker

ENTRYPOINT ["/bin/bash"]
