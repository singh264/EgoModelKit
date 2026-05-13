# EgoModelKit

EgoModelKit is a packaging and orchestration toolkit for reproducible egocentric-video model inference.

The first supported model will be runnable as:

```bash
egomodelkit run hand-object-contact \
  --input /path/to/image \
  --output /path/to/results
```

Docker execution is hidden behind the Python command-line interface (CLI).

## Development

Install the package locally in editable mode:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Run the automated quality checks:

```bash
ruff check .

pytest \
  --cov=egomodelkit \
  --cov-report=term-missing \
  --cov-fail-under=90

python -m build
```

Run a quick manual CLI smoke test:

```bash
egomodelkit --help
```

Expected rough output:
```text
Usage: egomodelkit [OPTIONS] COMMAND [ARGS]...

EgoModelKit: reproducible egocentric-video model packaging and inference.
```
