# Contributing

I ❤️ contributions! This project is built *on*, and *for*, open source. If you
would like to contribute, please read this document.


## 1. Reporting Issues

Report issues and bugs to the GitHub issue tracker for this project. I don't
have requirements for what must go in an issue, I trust use your judgement. At a
minimum, please include as much information as is required for me to reproduce
the issue, and copy and paste the output of:

```sh
$ bazel run //tools:whoami
```

If you can't use bazel, run the script directly and report its output:

```sh
$ ./tools/whoami.sh
```

If you find a bug, by far the best way to report it is to write a test which
exposes the issue and submit it as a PR. If you can do this, continue reading.


## 2. Contributing Code

This project is academic research code. Generally in academia, this means code
written to a poor standard and provided without tests or documentation. I don't
like this attitude. I am far from a perfect software engineer, but I strive to
write clean and maintainable code, and I am always learning. When writing new
code, use the existing sources as a lower bound on the quality you should be
aiming for. Specifically:


### 2.1. Tests

Please, please, test new code.

To paraphrase Kanye West, "one good test is worth a thousand patches". Use
bazel's excellent testing infrastructure to run the existing tests, and the new
ones that you write (see [INSTALL.md](/INSTALL.md) for instructions). Although
bazel caches test results and will perform only the minimum amount of testing on
incremental builds, this is a large project with a lot of tests, you may not
want to run all of them as many will not be relevant to what you are working on.
Use the CI script provided by bazel to run the tests on only those targets which
have been modified on your current branch:

```
$ ./third_party/bazel/ci.sh
```


## 2.2. Code Format

I have strong opinions about code formatting. Not in the specifics of how to
ident a certain block or where to put the parenthesis around a statement, but
rather that code formatting should be automated to cognitively offload the
developer to focus on more important things.

I wrote a tool, [format](https://github.com/ChrisCummins/format), which provides
automatic code formatters for most of the programming languages used in this
project. All checked code must pass through the formatters without modification.
Build and install the `format` binary into `~/.local/bin` using:

```sh
$ ./tools/format/install.sh
```

To run the formatter on a specific file, use:

``` sh
$ format <path...>
```

However, remembering to run a formatter every time you make an edit is a fool's
game. Instead, install the provided pre-commit hook to ensure that you never
miss a run:

```sh
$ format --install_pre_commit_hook
```

After installing the pre-commit hook, your commit messages will get a nice
shiny "Signed off by" footer. Please make sure you have this before submitting
a Pull Request.
