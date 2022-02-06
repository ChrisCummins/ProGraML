# Augment the standard base environment with Java.
FROM chriscummins/phd_base:2020.01.08
LABEL maintainer="Chris Cummins <chrisc.101@gmail.com>"

# Switch back to root user to do the installation admin.
USER root

# Workaround for broken JDK8 install.
# https://bugs.debian.org/cgi-bin/bugreport.cgi?bug=863199#23.
RUN mkdir -p /usr/share/man/man1

# Install Java.
RUN apt-get update \
  && apt-get install -y --no-install-recommends openjdk-11-jdk \
  && rm -rf /var/lib/apt/lists/*

# Switch back to user account.
USER docker

ENTRYPOINT ["/bin/bash"]
