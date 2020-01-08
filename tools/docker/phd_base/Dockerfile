# A minimal linux environment for bazel-built python binaries.
FROM python:3.7-slim
LABEL maintainer="Chris Cummins <chrisc.101@gmail.com>"

# Put Python into /bin, needed by bazel py_binary script entrypoints.
RUN ln -s /usr/local/bin/python /usr/bin/python \
  && ln -s /usr/local/bin/python /usr/bin/python3

# Install a few necessary bits and pieces:
RUN apt-get update && apt-get install -y --no-install-recommends \
    # A standard compiler toolchain which is needed for cldrive harnesses,
    # since the current clang compilation is not yet entirely hemetic and
    # relies on system linker and C headers:
    binutils \
    libgcc-7-dev \
    libc6-dev \
    # Providing libreadline.so which is needed by //third_party/oclgrind:
    libreadline6-dev \
    # Needed by tar to unpack .tar.bz2 files, which CLgen uses to encode corpus
    # archives:
    lbzip2 \
    # Needed by //third_party/py/git, and my own tools:
    git \
    # Needed by a handful of targets, e.g. //system/machines:
    rsync \
    # Required by //third_party/py/libxmp:
    libexempi-dev \
    # Needed by //system/machines:
    iputils-ping \
    # Needed because pip built //third_party/py/numpy is broken:
    python3-numpy \
    # Needed by //third_party/py/mysql:
    python3-dev \
    build-essential \
    default-libmysqlclient-dev \
    mysql-common \
    wget \
    # Tidy up:
    && rm -rf /var/lib/apt/lists/*

###############################################################################
# !!!DANGER ZONE!!! This section contains dirty hacks to workaround awkward
# dependencies with //third_party/py packages. Here be dragons!
#                __        _
#              _/  \    _(\(o
#             /     \  /  _  ^^^o
#            /   !   \/  ! '!!!v'
#           !  !  \ _' ( \____
#           ! . \ _!\   \===^\)
#            \ \_!  / __!
#         .   \!   /    \
#       (\_      _/   _\ )
#        \ ^^--^^ __-^ /(__
#         ^^----^^    "^--v'
#
# Dirty hack to install the version of libmysqlclient.so that MySQLdb python
# package is compiled against. This is fragile and will probably break when
# updating the base python image or version of mysqlclient package.
RUN wget http://launchpadlibrarian.net/449075198/libmysqlclient21_8.0.18-0ubuntu0.19.10.1_amd64.deb \
    && dpkg -i libmysqlclient21_8.0.18-0ubuntu0.19.10.1_amd64.deb \
    && rm libmysqlclient21_8.0.18-0ubuntu0.19.10.1_amd64.deb \
    # wget was only needed to download the libmysqlclient.so file above.
    && apt-get remove -y --purge wget
#
# Dirty hack to workaround the fact that oclgrind demands libreadline.so.6, but
# the current Ubuntu package provides libreadline.so.7.
RUN ln -s /lib/x86_64-linux-gnu/libreadline.so.7 \
    /lib/x86_64-linux-gnu/libreadline.so.6
#
# End of danger zone.
###############################################################################

# Remove unwanted files.
RUN rm -rf /usr/share/doc

# Create a non-root user to execute commands with. This is so that generated
# files in mapped volumes will be created as this user, rather than as root.
# Running as a non-priveledges user may be iritating for debugging, in which
# case pass the `-u root` flag to your docker command.
RUN useradd -ms /bin/bash docker
# Add docker to a 'other_docker' group, which enables read permission when
# mapping /var/run/docker.sock to enable nested docker.
RUN groupadd --gid 123 other_docker && usermod -a -G other_docker docker
USER docker

ENTRYPOINT ["/bin/bash"]
