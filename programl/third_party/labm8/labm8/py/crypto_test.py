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
"""Unit tests for //labm8/py:crypto."""
from labm8.py import app, crypto, test

FLAGS = app.FLAGS


# sha1()
def test_sha1_empty_str():
    assert "da39a3ee5e6b4b0d3255bfef95601890afd80709" == crypto.sha1_str("")


def test_sha1_hello_world():
    assert "0a0a9f2a6772942557ab5355d76af442f8f65e01" == crypto.sha1_str(
        "Hello, World!",
    )


# sha1_list()
def test_sha1_empty_list():
    assert "da39a3ee5e6b4b0d3255bfef95601890afd80709" == crypto.sha1_list()
    assert "97d170e1550eee4afc0af065b78cda302a97674c" == crypto.sha1_list([])


def test_sha1_list():
    assert "06bf71070d31b2ebe4bdae828fc76a70e4b56f00" == crypto.sha1_list(
        ["hello", "world"],
    )


# sha1_file()
def test_sha1_file_empty():
    assert "da39a3ee5e6b4b0d3255bfef95601890afd80709" == crypto.sha1_file(
        "labm8/py/test_data/empty_file",
    )


def test_sha1_file_hello_world():
    assert "09fac8dbfd27bd9b4d23a00eb648aa751789536d" == crypto.sha1_file(
        "labm8/py/test_data/hello_world",
    )


# md5()
def test_md5_empty_str():
    assert "d41d8cd98f00b204e9800998ecf8427e" == crypto.md5_str("")


def test_md5_hello_world():
    assert "65a8e27d8879283831b664bd8b7f0ad4" == crypto.md5_str("Hello, World!")


# md5_list()
def test_md5_empty_list():
    assert "d41d8cd98f00b204e9800998ecf8427e" == crypto.md5_list()
    assert "d751713988987e9331980363e24189ce" == crypto.md5_list([])


def test_md5_list():
    assert "6ded24f0b2f43dd31e601a27fcecb7e8" == crypto.md5_list(
        ["hello", "world"],
    )


# md5_file()
def test_md5_file_empty():
    assert "d41d8cd98f00b204e9800998ecf8427e" == crypto.md5_file(
        "labm8/py/test_data/empty_file",
    )


def test_md5_file_hello_world():
    assert "746308829575e17c3331bbcb00c0898b" == crypto.md5_file(
        "labm8/py/test_data/hello_world",
    )


# sha256()
def test_sha256_empty_str():
    assert (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        == crypto.sha256_str("")
    )


def test_sha256_hello_world():
    assert (
        "dffd6021bb2bd5b0af676290809ec3a53191dd81c7f70a4b28688a362182986f"
        == crypto.sha256_str("Hello, World!")
    )


# sha256_list()
def test_sha256_empty_list():
    assert (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        == crypto.sha256_list()
    )
    assert (
        "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"
        == crypto.sha256_list([])
    )


def test_sha256_list():
    assert (
        "be3d036085587af9522a8358dd1d09ba2b0ec63db92a62d28cf00dfcaeb25ca1"
        == crypto.sha256_list(["hello", "world"])
    )


# sha256_file()
def test_sha256_file_empty():
    assert (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        == crypto.sha256_file("labm8/py/test_data/empty_file")
    )


def test_sha256_file_hello_world():
    assert (
        "d9014c4624844aa5bac314773d6b689ad467fa4e1d1a50a1b8a99d5a95f72ff5"
        == crypto.sha256_file("labm8/py/test_data/hello_world")
    )


if __name__ == "__main__":
    test.Main()
