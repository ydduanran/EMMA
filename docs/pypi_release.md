# PyPI Release

This repository is already structured as an installable Python package.

## Install From GitHub

```bash
pip install "git+https://github.com/ydduanran/EMMA.git"
```

## Build Local Distributions

```bash
pip install build twine
python -m build
```

This creates:

```text
dist/emma_3dgenome-0.2.1.tar.gz
dist/emma_3dgenome-0.2.1-py3-none-any.whl
```

## Check The Package

```bash
python -m twine check dist/*
```

## Upload To PyPI

Use a PyPI API token instead of a password:

```bash
python -m twine upload dist/*
```

After upload, users can install with:

```bash
pip install emma-3dgenome
```

## Version Update

Before each new release, update the version in:

```text
pyproject.toml
src/emma_3dgenome/__init__.py
```
