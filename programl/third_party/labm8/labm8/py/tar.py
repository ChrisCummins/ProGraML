# Copyright 2014-2020 Chris Cummins <chrisc.101@gmail.com>.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tarball util.
"""
import tarfile

from labm8.py import fs


def unpack_archive(*components, **kwargs) -> str:
    """
    Unpack a compressed archive.

    Arguments:
        *components (str[]): Absolute path.
        **kwargs (dict, optional): Set "compression" to compression type.
            Default: bz2. Set "dir" to destination directory. Defaults to the
            directory of the archive.

    Returns:
        str: Path to directory.
    """
    path = fs.abspath(*components)
    compression = kwargs.get("compression", "bz2")
    dir = kwargs.get("dir", fs.dirname(path))

    fs.cd(dir)
    tar = tarfile.open(path, "r:" + compression)
    tar.extractall()
    tar.close()
    fs.cdpop()

    return dir
