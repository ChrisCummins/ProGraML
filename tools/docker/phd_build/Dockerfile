# An Ubuntu environment configured for building the phd repo.
FROM chriscummins/phd_base:latest
MAINTAINER Chris Cummins <chrisc.101@gmail.com>

# Switch back to root user to do the installation admin.
USER root

# Disable post-install interactive configuration.
# For example, the package tzdata runs a post-installation prompt to select the
# timezone.
ENV DEBIAN_FRONTEND noninteractive

# Install packages required to install bazel.
RUN apt-get update
RUN apt-get install -y curl unzip

# Install bazel.
RUN curl -L -o /tmp/bazel.sh https://github.com/bazelbuild/bazel/releases/download/0.20.0/bazel-0.20.0-installer-linux-x86_64.sh && \
    bash /tmp/bazel.sh && rm /tmp/bazel.sh

# Workaround for broken JDK8 install.
# https://bugs.debian.org/cgi-bin/bugreport.cgi?bug=863199#23.
RUN mkdir -p /usr/share/man/man1

# Install packages required to build the code.
RUN apt-get install -y default-libmysqlclient-dev g++ zlib1g-dev \
  ocl-icd-opencl-dev python openjdk-8-jdk apt-transport-https \
  ca-certificates gnupg-agent software-properties-common texlive-full rsync

# Install docker (needed to build docker images).
RUN curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add - \
    && add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu xenial stable" \
    && apt-get update
RUN apt-get install -y docker-ce docker-ce-cli containerd.io

RUN mkdir -pv /phd && chown docker:docker -R /phd
WORKDIR /phd
USER docker

# For compiling arduino code.
RUN python -m pip install platformio

# Needed for compatability with broken python 3 support.
ENV BAZEL_PYTHON /phd/tools/py3_wrapper.sh

ENTRYPOINT ["/bin/bash"]
