# Augment the standard base environment with Java.
FROM chriscummins/phd_base_java:2020.01.08
LABEL maintainer="Chris Cummins <chrisc.101@gmail.com>"

# Switch back to root user to do the installation admin.
USER root

# Install Tensorflow because the bazel pip_import dependency at
# //third_party/py/tensorflow is broken.
RUN python3 -m pip install 'tensorflow==1.14.0'

# Switch back to user account.
USER docker

ENTRYPOINT ["/bin/bash"]
