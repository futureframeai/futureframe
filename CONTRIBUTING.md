# Contributing

## Development

### Nightly Install

Install Python3 and clone the repository.

```bash
git clone https://github.com/futureframeai/futureframe.git
cd futureframe
```

Install Poetry in a Python virtual environment.

```bash
VENV_PATH=venv
python3 -m venv $VENV_PATH
$VENV_PATH/bin/pip install -U pip setuptools
$VENV_PATH/bin/pip install poetry
poetry shell
```

Install the latest version from the source.

```bash
poetry install
```

### Scripts

Format:

```bash
poetry run ruff format futureframe examples tests scripts
```

Fix:

```bash
poetry run ruff check format futureframe examples tests --fix
```

Test:

```bash
poetry run pytest -n auto -v -s --cov
```

Build:

```bash
poetry build
```

Valid version bump rules:

```bash
# patch, minor, major, prepatch, preminor, premajor, prerelease.
poetry version <bump>
```

Publish:

```bash
# poetry config pypi-token.pypi your-token
# poetry publish --username myprivaterepo --password <password> --repository myprivaterepo
poetry publish
```

Docs:

```bash
# Dev
poetry run mkdocs serve
# Build
python scripts/gen_docs_index.py
python scripts/gen_docs_ref_pages.py
poetry run mkdocs build -d dist/docs
# Publish
# poetry run mkdocs gh-deploy -d dist/docs -b docs
```

Docstrings:

We use Google-style docstrings.

```bash
# Test docstring examples
python -m doctest -v futureframe/predict.py
```
