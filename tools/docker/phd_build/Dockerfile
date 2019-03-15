# An Ubuntu environment configured for building the phd repo.
FROM chriscummins/phd_base
MAINTAINER Chris Cummins <chrisc.101@gmail.com>

# Disable post-install interactive configuration.
# For example, the package tzdata runs a post-installation prompt to select the
# timezone.
ENV DEBIAN_FRONTEND noninteractive

# Setup the environment.
ENV HOME /root
ENV USER docker
ENV PHD /phd

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
  ca-certificates gnupg-agent software-properties-common

# Install docker (needed to build docker images).
RUN curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add - \
    && add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu xenial stable" \
    && apt-get update
RUN apt-get install -y docker-ce docker-ce-cli containerd.io

# Install my preferred development tools.
RUN apt-get install -y zsh vim

# Clone the source code for this project.
RUN apt-get install -y git
RUN git clone https://github.com/ChrisCummins/phd.git $PHD

WORKDIR $PHD

# Configure the project.
RUN ./configure --noninteractive

# Install my Zsh and Vim configs.
RUN $PHD/system/dotfiles/run -v Zsh Vim

CMD ["/bin/zsh"]
