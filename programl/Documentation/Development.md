# Development

This document describes the development workflow for ProGraML.

## Branching Model

This project uses the
[git flow](https://danielkummer.github.io/git-flow-cheatsheet/) branching model
to ensure that changes to the graph data format which break compatibility are
recognized by different release versions.

To use git flow:

1. [Install git-flow](https://github.com/nvie/gitflow/wiki/Installation).
2. Initialize it in this repository by running:

```
$ git flow init

Which branch should be used for bringing forth production releases?
Branch name for production releases: stable

Which branch should be used for integration of the "next release"?
Branch name for "next release" development: development

How to name your supporting branch prefixes?
Feature branches? [feature/]
Release branches? [release/]
Hotfix branches? [hotfix/]
Support branches? [support/]
Version tag prefix? []
```
