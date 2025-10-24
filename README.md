# test-tensors

[![License](https://img.shields.io/pypi/l/test-tensors.svg?color=green)](https://github.com/jojoelfe/test-tensors/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/test-tensors.svg?color=green)](https://pypi.org/project/test-tensors)
[![Python Version](https://img.shields.io/pypi/pyversions/test-tensors.svg?color=green)](https://python.org)
[![CI](https://github.com/jojoelfe/test-tensors/actions/workflows/ci.yml/badge.svg)](https://github.com/jojoelfe/test-tensors/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/jojoelfe/test-tensors/branch/main/graph/badge.svg)](https://codecov.io/gh/jojoelfe/test-tensors)

Provides simply tensors for testing tensor manipulation algorithms

## Development

The easiest way to get started is to use the [github cli](https://cli.github.com)
and [uv](https://docs.astral.sh/uv/getting-started/installation/):

```sh
gh repo fork jojoelfe/test-tensors --clone
# or just
# gh repo clone jojoelfe/test-tensors
cd test-tensors
uv sync
```

Run tests:

```sh
uv run pytest
```

Lint files:

```sh
uv run pre-commit run --all-files
```
