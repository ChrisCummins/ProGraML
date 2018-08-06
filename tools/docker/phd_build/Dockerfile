# An Ubuntu environment configured for building the phd repo.
FROM ubuntu:18.04
MAINTAINER Chris Cummins <chrisc.101@gmail.com>

# Disable post-install interactive configuration.
# For example, the package tzdata runs a post-installation prompt to select the
# timezone.
ENV DEBIAN_FRONTEND noninteractive

# Setup the environment.
ENV HOME /root
ENV USER docker
ENV PHD /phd

# Install essential packages.
RUN apt-get update
RUN apt-get install --no-install-recommends -y \
    ca-certificates curl g++ git libmysqlclient-dev ocl-icd-opencl-dev \
    pkg-config python python-dev python3.6 python3.6-dev python3-distutils \
    unzip zip zlib1g-dev openjdk-11-jdk m4 libexempi-dev rsync texlive-full \
    python3-numpy

# Install bazel.
RUN curl -L -o /tmp/bazel.sh https://github.com/bazelbuild/bazel/releases/download/0.14.1/bazel-0.14.1-installer-linux-x86_64.sh && \
    bash /tmp/bazel.sh && rm /tmp/bazel.sh

# Clone the source code for this project.
RUN git clone https://github.com/ChrisCummins/phd.git $PHD

WORKDIR $PHD

# Configure the project.
RUN ./configure --noninteractive

# Ensure that new shells use the build env.
RUN echo "source $PHD/.env" >> ~/.bashrc

# Run a py_test target so that the bazel installation is unpacked and python
# packages are installed.
RUN . $PHD/.env && bazel test //config:getconfig_test

CMD ["/bin/bash"]
