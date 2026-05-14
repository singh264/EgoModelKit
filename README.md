# EgoModelKit

EgoModelKit is a packaging and orchestration toolkit for reproducible egocentric-video model inference.

The first supported model will be runnable as:

```bash
egomodelkit run hand-object-contact \
  --input /path/to/image \
  --output /path/to/results
```

The public interface is run-only. Runtime preparation, container use, and model environment details stay hidden behind the command-line interface (CLI).

## Current Milestone

The public Shan command shape is now available.

At this commit, use --dry-run to validate a request:

```bash
egomodelkit run hand-object-contact \
  --input /path/to/image.jpg \
  --output /path/to/results \
  --dry-run
```

Expected output:

```text
Dry run: hand-object-contact request is valid.
Input: /path/to/image.jpg
Output: /path/to/results
```

## Development

Clone the repository, then create and activate a virtual environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

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

Expected output:

```text
Usage: egomodelkit [OPTIONS] COMMAND [ARGS]...

EgoModelKit: reproducible egocentric-video model packaging and inference.
```
