# Contributing

We ❤️ contributions! This project is built *on*, and *for*, open source. If you
would like to contribute, please read this document.


## Reporting Issues

Report issues and bugs to the GitHub issue tracker for this project. We don't
have requirements for what must go in an issue, I trust your judgment. Please
include as much information as is required for us to reproduce the issue, and
copy and paste the output of:

```sh
$ ./tools/whoami.sh
```

If you find a bug, by far the best way to report it is to write a test which
exposes the issue and submit it as a PR. If you can do this, continue reading.


## Contributing Code

To contribute to this project:

1. Fork the repo and create your branch from development.
2. Follow the instructions for building from source to set up your environment.
3. If you've added code that should be tested, add tests.
4. If you've changed APIs, update the documentation.
5. Ensure the make test suite passes.
6. Make sure your code lints (see Code Style below).
7. If you haven't already, complete the Contributor License Agreement ("CLA").


## Code Style

Our code style is simple:

* Python:
  [black](https://github.com/psf/black/blob/master/docs/the_black_code_style.md)
  and [isort](https://pypi.org/project/isort/).
* C++: [Google C++
  style](https://google.github.io/styleguide/cppguide.html) with 100
  character line length and `camelCaseFunctionNames()`.

We use [pre-commit](https://pre-commit.com/) to ensure that code is formatted
prior to committing. Before submitting pull requests, please run pre-commit. See
the [config file](/.pre-commit-config.yaml) for installation and usage
instructions.

Other common sense rules we encourage are:

* Prefer descriptive names over short ones.
* Split complex code into small units.
* When writing new features, add tests.
* Make tests deterministic.
* Prefer easy-to-use code over easy-to-read, and easy-to-read code over
  easy-to-write.
