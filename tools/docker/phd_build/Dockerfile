# An Ubuntu environment configured for building the phd repo.
FROM chriscummins/phd_base_java:latest
MAINTAINER Chris Cummins <chrisc.101@gmail.com>

# Switch back to root user to do the installation admin.
USER root

# Disable post-install interactive configuration.
# For example, the package tzdata runs a post-installation prompt to select the
# timezone.
ENV DEBIAN_FRONTEND noninteractive

# Install packages required to build the code.
RUN apt-get update && apt-get install -y --no-install-recommends \
  curl unzip default-libmysqlclient-dev g++ zlib1g-dev ocl-icd-opencl-dev python \
  apt-transport-https ca-certificates gnupg-agent \
  software-properties-common texlive-full rsync

# Install docker (needed to build docker images).
RUN apt-get update && apt-get install -y gnupg2 \
    && curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add - \
    && add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu xenial stable"
RUN apt-get update && apt-get install -y docker-ce docker-ce-cli containerd.io

# Install golang.
RUN curl https://dl.google.com/go/go1.12.6.linux-amd64.tar.gz > /tmp/go.tar.gz \
    && tar -C /usr/local -xzf /tmp/go.tar.gz \
    && rm /tmp/go.tar.gz
ENV GOPATH /home/docker/go
ENV PATH /usr/local/go/bin:/home/docker/go/bin:$PATH

# Install nodejs.
RUN curl -sL https://deb.nodesource.com/setup_10.x | sudo -E bash - \
    && apt-get update && apt-get install -y nodejs && npm -v

# Install linters.
RUN apt update && apt-get install -y --no-install-recommends \
    clang-format \
    && go get github.com/bazelbuild/buildtools/buildifier \
    && go get github.com/uber/prototool/cmd/prototool \
    && python -m pip install yapf sqlparse \
    && npm install -g js-beautify jsonlint \
    && wget https://github.com/mvdan/sh/releases/download/v2.6.4/shfmt_v2.6.4_linux_386 -O /usr/local/bin/shfmt \
    && chmod +x /usr/local/bin/shfmt

# Install bazel.
ENV BAZEL_VERSION 0.28.1
RUN curl -L -o /tmp/bazel.sh https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    bash /tmp/bazel.sh && rm /tmp/bazel.sh

# For compiling arduino code.
# We need to use python2 because of the time of writing:
#   PlatformIO Core v3.6.7 does not run under Python version 3.6.8 (default,
#       Mar  4 2019, 23:39:18)
#   [GCC 6.3.0 20170516].
#   Minimum supported version is 2.7, please upgrade Python.
#   Python 3 is not yet supported.
RUN apt-get update && apt-get install -y python-pip
RUN pip2 install platformio

RUN apt-get install libmariadbclient18

# Install a text editor.
RUN apt-get update && apt-get install -y --no-install-recommends vim

RUN mkdir -pv /phd && chown docker:docker -R /phd
WORKDIR /phd
USER docker

# Needed for compatability with broken python 3 support.
ENV BAZEL_PYTHON /phd/tools/py3_wrapper.sh

ENTRYPOINT ["/bin/bash"]
