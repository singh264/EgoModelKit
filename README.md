# EgoModelKit

EgoModelKit is a packaging and orchestration toolkit for reproducible egocentric-video model inference.

The first supported model is Shan's hand-object-contact detector:

```bash
egomodelkit run hand-object-contact \
  --input /path/to/image \
  --output /path/to/results
```

The public interface is run-only. Runtime preparation, container use, and model environment details stay hidden behind the command-line interface (CLI).

## Current Milestone

The Shan hand-object-contact command now performs real inference.

Run:

```bash
egomodelkit run hand-object-contact \
  --input /path/to/image.jpg \
  --output /path/to/results
```

On the first real run, EgoModelKit prepares the hidden runtime automatically if it is not already available. Later runs reuse that prepared runtime.

Expected output:

```text
Completed: hand-object-contact
Outputs: /path/to/results
```

A successful run should write Shan visualization output into the requested results directory, typically ending in:

```text
_det.png
```

## Optional Dry Run

Use `--dry-run` to validate a request without executing inference:

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

## Current Platform Target

The current target is a research-group Linux NVIDIA GPU machine. EgoModelKit hides container execution, but the underlying runtime still depends on GPU-enabled container support. The public command remains intentionally stable and operating-system-neutral so the runtime layer can later be adjusted for additional lab environments without changing the CLI.


## Development

Clone the repository, then create and activate a virtual environment:

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
