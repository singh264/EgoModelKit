# EgoModelKit

EgoModelKit is a packaging and orchestration toolkit for reproducible egocentric-video model inference.

The currently supported model adapters are:

<details>
<summary><strong>hand-object-contact</strong></summary>

Shan's hand-object-contact detector can run on either one supported image file or a directory of supported image files:

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

`--input` may be either:

- one supported image file, or
- a directory containing one or more supported image files.

Supported image suffixes:

```text
.jpg, .jpeg, .png, .bmp, .webp
```

Directory input is processed non-recursively in the current milestone.

</details>

<details>
<summary><strong>adl-recognition</strong></summary>

Adesh's ADL recognition adapter runs an EgoVizML-based activity recognition flow that uses video extraction, hand-object-contact outputs, Detic outputs, and a final ADL classifier:

```bash
egomodelkit run adl-recognition \
  --input /path/to/video \
  --output /path/to/results
```

```bash
egomodelkit run adl-recognition \
  --input /path/to/video-directory \
  --output /path/to/results
```

`--input` may be either:

- one MP4 video file,
- a directory containing one or more MP4 video files, or
- an existing `all_preds.pkl` file for prediction-only reuse.

Directory input is processed non-recursively in the current milestone.

</details>

The public interfaces are the `run` command for direct CLI execution and the `gui` command for launching the local browser GUI. Runtime preparation, container use, external model repositories, model checkpoints, and model environment details stay hidden behind these interfaces.

Model code and checkpoint provenance is centralized in `src/egomodelkit/runtime/external_code.py`. The hidden runtime images are built from pinned project-controlled fork URLs and pinned checkpoint sources so runs can be audited and reproduced.

## Current Milestone

The `run` command now performs real inference for both supported model adapters.

<details>
<summary><strong>hand-object-contact milestone</strong></summary>

The `hand-object-contact` command performs real inference for either one supported image file or a directory of supported image files.

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

For directory input, EgoModelKit writes one visualization image, one JSON file, and one pickle file for each processed input image stem.

The JSON file provides a portable structured representation of detected hands and objects. The pickle file preserves Python-friendly raw prediction structures for downstream research workflows.

The hidden hand-object-contact runtime is built from an EgoModelKit-maintained fork of the original hand-object-detector repository, pinned to a specific commit. This allows EgoModelKit to preserve the upstream model behavior while adding packaging-oriented outputs such as JSON and pickle prediction files.

</details>

<details>
<summary><strong>adl-recognition milestone</strong></summary>

The `adl-recognition` command now performs packaged ADL inference with hidden EgoVizML, hand-object-contact, Detic, and Detectron2 orchestration.

Run from video:

```bash
egomodelkit run adl-recognition \
  --input /path/to/video.mp4 \
  --output /path/to/results
```

Run from a directory of videos:

```bash
egomodelkit run adl-recognition \
  --input /path/to/video-directory \
  --output /path/to/results
```

Run from an existing EgoVizML combined prediction file:

```bash
egomodelkit run adl-recognition \
  --input /path/to/all_preds.pkl \
  --output /path/to/results
```

On the first real run, EgoModelKit prepares the hidden runtime images automatically if they are not already available. Later runs reuse those prepared runtimes.

Expected output:

```text
Completed: adl-recognition
Outputs: /path/to/results
```

A successful full video run should write final outputs into the requested results directory, typically including:

```text
all_preds.pkl
adl_predictions.csv
adl_predictions_summary.csv
adl_recognition_work/
```

`all_preds.pkl` is the combined EgoVizML-style prediction file. `adl_predictions.csv` contains the full classifier output. `adl_predictions_summary.csv` contains a smaller summary intended for quick review. `adl_recognition_work/` contains intermediate staged videos, extracted frames, and model-stage outputs.

A prediction-only run from an existing `all_preds.pkl` writes:

```text
adl_predictions.csv
adl_predictions_summary.csv
```

The hidden ADL runtime is built from pinned external code revisions for EgoVizML, Detic and Detectron2. This allows EgoModelKit to preserve comparable model behavior while hiding the multi-repository runtime orchestration behind one CLI command.

</details>

## Runtime Checks and Progress Messages

Before executing inference, the `run` command checks the host prerequisites needed by the current runtime and reports progress clearly.

<details>
<summary><strong>hand-object-contact example output</strong></summary>

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

</details>

<details>
<summary><strong>adl-recognition example output</strong></summary>

Full video run:

```text
EgoModelKit: Validating adl-recognition request.
EgoModelKit: Checking host runtime prerequisites.
EgoModelKit: Python 3.12.2 detected.
EgoModelKit: Docker executable found: /usr/bin/docker
EgoModelKit: Docker daemon is available.
EgoModelKit: Using output directory: /path/to/results
EgoModelKit: Checking packaged adl-recognition core runtime image.
EgoModelKit: Packaged adl-recognition core runtime image is already available.
EgoModelKit: Starting ADL video frame extraction.
EgoModelKit runtime: preparing EgoVizML video workspace.
EgoModelKit runtime: calling EgoVizML frame extraction.
EgoModelKit runtime: EgoVizML frame extraction finished.
EgoModelKit: Finished ADL video frame extraction.
EgoModelKit: Checking packaged adl-recognition Detic runtime image.
EgoModelKit: Packaged adl-recognition Detic runtime image is already available.
EgoModelKit: Validating hand-object-contact request.
EgoModelKit: Checking host runtime prerequisites.
EgoModelKit: Python 3.12.2 detected.
EgoModelKit: Docker executable found: /usr/bin/docker
EgoModelKit: Docker daemon is available.
EgoModelKit: Using output directory: /path/to/results/adl_recognition_work/egoviz_data/meal-preparation-cleanup/subclips_shan/video001--1
EgoModelKit: Checking packaged hand-object-contact runtime image.
EgoModelKit: Packaged hand-object-contact runtime image is already available.
EgoModelKit: Starting hand-object-contact inference.
EgoModelKit runtime: preparing output directory.
EgoModelKit runtime: staging input image(s) for hand-object-contact.
EgoModelKit runtime: launching hand-object-contact demo inference.
EgoModelKit runtime: hand-object-contact inference finished.
EgoModelKit: hand-object-contact inference completed.
EgoModelKit: Starting Detic inference for video001--1.
EgoModelKit: Finished Detic inference for video001--1.
EgoModelKit: Starting ADL prediction finalization.
EgoModelKit runtime: preparing EgoVizML prediction folders.
EgoModelKit runtime: paired 1 Detic/Shan frame outputs.
EgoModelKit runtime: calling EgoVizML process_all_preds.py
EgoModelKit runtime: calling EgoVizML classifier wrapper.
Saved predictions to: /workspace/output/adl_predictions.csv
Saved prediction summary to: /workspace/output/adl_predictions_summary.csv
EgoModelKit runtime: ADL prediction outputs are ready.
EgoModelKit: Finished ADL prediction finalization.
EgoModelKit: adl-recognition inference completed.
Completed: adl-recognition
Outputs: /path/to/results
```

For longer videos, the hand-object-contact and Detic stage messages repeat for each extracted subclip frame directory.

Prediction-only run from an existing `all_preds.pkl`:

```text
EgoModelKit: Validating adl-recognition request.
EgoModelKit: Checking host runtime prerequisites.
EgoModelKit: Python 3.12.2 detected.
EgoModelKit: Docker executable found: /usr/bin/docker
EgoModelKit: Docker daemon is available.
EgoModelKit: Using output directory: /path/to/results
EgoModelKit: Checking packaged adl-recognition core runtime image.
EgoModelKit: Packaged adl-recognition core runtime image is already available.
EgoModelKit: Starting adl-recognition prediction.
EgoModelKit runtime: calling EgoVizML classifier wrapper.
Saved predictions to: /workspace/output/adl_predictions.csv
Saved prediction summary to: /workspace/output/adl_predictions_summary.csv
EgoModelKit runtime: ADL prediction outputs are ready.
EgoModelKit: Finished adl-recognition prediction.
EgoModelKit: adl-recognition inference completed.
Completed: adl-recognition
Outputs: /path/to/results
```

</details>

## Optional Dry Run

Use `--dry-run` to validate a request without executing inference:

<details>
<summary><strong>hand-object-contact dry run</strong></summary>

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

</details>

<details>
<summary><strong>adl-recognition dry run</strong></summary>

```bash
egomodelkit run adl-recognition \
  --input /path/to/video.mp4 \
  --output /path/to/results \
  --dry-run
```

Expected output:

```text
Dry run: adl-recognition request is valid.
Input: /path/to/video.mp4
Output: /path/to/results
```

The same dry-run flow also accepts a directory input that contains one or more supported video files, or an existing `all_preds.pkl` file.

</details>

## Local Browser GUI

EgoModelKit also provides a local browser GUI backend for users who should not need to run model commands directly.

Install the optional GUI dependencies:

```bash
python -m pip install -e ".[gui]"
```

Launch the GUI:

```bash
egomodelkit gui
```

Launch without automatically opening a browser window:

```bash
egomodelkit gui --no-browser
```

Use a custom local port:

```bash
egomodelkit gui --port 7860 --no-browser
```

The GUI server binds to `127.0.0.1` by default. This keeps the interface local-only for privacy-sensitive research workflows. The backend exposes a small local API for the React frontend:

```text
GET  /api/models
POST /api/output-preview
POST /api/dry-run
POST /api/runs
GET  /api/runs/{run_id}/progress
POST /api/open-output-folder
POST /api/select-output-folder
```

During backend development, the API documentation is available at:

```text
http://127.0.0.1:7860/docs
```

The React frontend build is served from `src/egomodelkit/web/dist` when those frontend assets are present. The frontend source/build artifacts are tracked separately from the backend GUI API work.

Frontend development files live in:

```text
src/egomodelkit/web
```

Run the frontend locally during development:

```bash
cd src/egomodelkit/web
npm ci
npm run dev
```

Run frontend checks:

```bash
npm run typecheck
npm test
npm run build
```

The frontend uses Vite, React, Tailwind CSS, Vitest, and Testing Library. The implemented GUI flow currently includes the welcome/start screen, backend-loaded model-selection screen, single-file and multi-file input selection with backend-provided model-specific file filtering, output-folder selection screen, review screen, dry-run action with local Linux/NVIDIA runtime checks, run progress polling, a fixed-size scrollable progress log, completed/failed results screens, an output-folder preview screen available after a completed run, and open-output-folder actions. The output folder selected in the GUI is treated as the output root; each run writes its files under a generated `run-*` subfolder inside that root, with runtime outputs normalized into `results/`, `visual_outputs/`, `technical/`, and `logs/` folders. ADL runs include stub video-level metric CSV files until the Bandini-style metric implementation is added.

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
python -m pip install -e ".[dev,gui]"
```

The test suite includes GUI backend tests. Use `.[dev,gui]` for local development so FastAPI, Uvicorn, multipart upload handling, and HTTP test dependencies are available.

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
