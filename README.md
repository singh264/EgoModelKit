# EgoModelKit

EgoModelKit is a packaging and orchestration toolkit for reproducible egocentric-video model inference.

The first supported model is Shan's hand-object-contact detector:

```bash
egomodelkit run hand-object-contact \
  --input /path/to/image \
  --output /path/to/results
```

```bash
egomodelkit run hand-object-contact \
  --input /path/to/image-directory \
  --output /path/to/results
```

The public interface is run-only. Runtime preparation, container use, and model environment details stay hidden behind the command-line interface (CLI).

`--input` may be either:

- one supported image file, or
- a directory containing one or more supported image files.

Directory input is processed non-recursively in the current milestone.

## Current Milestone

The hand-object-contact command now performs real inference for either one supported image file or a directory of supported image files.

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

A successful run should write visualization output into the requested results directory, typically including:

```text
<image_stem>_det.png
<image_stem>_shan.json
<image_stem>_shan.pkl
```

The JSON file provides a portable structured representation of detected hands and objects. The pickle file preserves Python-friendly raw prediction structures for downstream research workflows.

The hidden hand-object-contact runtime is built from an EgoModelKit-maintained fork of the original hand-object-detector repository, pinned to a specific commit. This allows EgoModelKit to preserve the upstream model behavior while adding packaging-oriented outputs such as JSON and pickle prediction files.

## Runtime Checks and Progress Messages

Before executing inference, the `run` command checks the host prerequisites needed by the current runtime and reports progress clearly.

Example output:

```text
EgoModelKit: Validating hand-object-contact request.
EgoModelKit: Checking host runtime prerequisites.
EgoModelKit: Python 3.12.2 detected.
EgoModelKit: Docker executable found: /usr/bin/docker
EgoModelKit: Docker daemon is available.
EgoModelKit: Using output directory: /path/to/results
EgoModelKit: Checking packaged hand-object-contact runtime image.
EgoModelKit: Packaged hand-object-contact runtime image is already available.
EgoModelKit: Starting hand-object-contact inference.
EgoModelKit runtime: preparing output directory.
EgoModelKit runtime: staging input image(s) for hand-object-contact.
EgoModelKit runtime: launching hand-object-contact demo inference.
EgoModelKit runtime: hand-object-contact inference finished.
EgoModelKit: hand-object-contact inference completed.
Completed: hand-object-contact
Outputs: /path/to/results
```

For directory input, EgoModelKit writes one visualization image, one JSON file, and one pickle file for each processed input image stem.

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

The same dry-run flow also accepts a directory input that contains one or more supported image files.

## Current Platform Target

The current target is a research-group Linux NVIDIA GPU machine. EgoModelKit hides container execution, but the underlying runtime still depends on GPU-enabled container support. The public command remains intentionally stable and operating-system-neutral so the runtime layer can later be adjusted for additional lab environments without changing the CLI.


## Development

Clone the repository, then create and activate a virtual environment. Use Python 3.10 or newer for the EgoModelKit development environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
python --version
```

The final `python --version` check should report Python 3.10 or newer.

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
