# An Ubuntu environment configured for building the phd repo.
# TODO(github.com/ChrisCummins/phd/issues/63): Clone phd repo, run configure, 
# run bootstrap, remove phd clone.
FROM chriscummins/phd_base_tf_cpu:2020.01.08
LABEL maintainer="Chris Cummins <chrisc.101@gmail.com>"

# Switch back to root user to do the installation admin.
USER root

# Disable post-install interactive configuration.
# For example, the package tzdata runs a post-installation prompt to select the
# timezone.
ENV DEBIAN_FRONTEND noninteractive

# Install packages.
RUN apt-get update && apt-get install -y --no-install-recommends \
  # Packages required to build code.
  unzip \
  g++ \
  zlib1g-dev \
  ocl-icd-opencl-dev \
  python \
  apt-transport-https \
  ca-certificates gnupg-agent \
  software-properties-common \
  texlive-full \
  patch \
  zip \
  m4 \
  # Docker
  docker.io \
  # Tidy up.
  && rm -rf /var/lib/apt/lists/*

# Install golang.
RUN apt update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*
RUN curl https://dl.google.com/go/go1.12.6.linux-amd64.tar.gz > /tmp/go.tar.gz \
    && tar -C /usr/local -xzf /tmp/go.tar.gz \
    && rm /tmp/go.tar.gz
ENV GOPATH /home/docker/go
ENV PATH /usr/local/go/bin:/home/docker/go/bin:$PATH

# Install nodejs.
RUN curl -sL https://deb.nodesource.com/setup_10.x | bash - \
    && apt-get update \
    && apt-get install -y --no-install-recommends nodejs && npm -v \
    && rm -rf /var/lib/apt/lists/*

# Install linters.
RUN apt update \
    && apt-get install -y --no-install-recommends clang-format wget \
    && go get github.com/bazelbuild/buildtools/buildifier \
    && go get github.com/uber/prototool/cmd/prototool \
    && python -m pip install sqlparse \
    && npm install -g js-beautify jsonlint \
    && wget https://github.com/mvdan/sh/releases/download/v2.6.4/shfmt_v2.6.4_linux_386 -O /usr/local/bin/shfmt \
    && chmod +x /usr/local/bin/shfmt \
    && apt-get remove -y wget \
    && rm -rf /var/lib/apt/lists/*

# Install bazel.
ENV BAZEL_VERSION 2.0.0
RUN curl -L -o /tmp/bazel.sh https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    bash /tmp/bazel.sh && rm /tmp/bazel.sh

# For compiling arduino code.
# We need to use python2 because of the time of writing:
#   PlatformIO Core v3.6.7 does not run under Python version 3.6.8 (default,
#       Mar  4 2019, 23:39:18)
#   [GCC 6.3.0 20170516].
#   Minimum supported version is 2.7, please upgrade Python.
#   Python 3 is not yet supported.
RUN apt-get update \
    && apt-get install -y --no-install-recommends python-pip python-setuptools \
    && rm -rf /var/lib/apt/lists/*
RUN pip2 install platformio

# Install a text editor.
RUN apt-get update && apt-get install -y --no-install-recommends vim \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -pv /phd && chown docker:docker -R /phd
WORKDIR /phd
USER docker

# Needed for compatability with broken python 3 support.
ENV BAZEL_PYTHON /phd/tools/py3_wrapper.sh

ENTRYPOINT ["/bin/bash"]
