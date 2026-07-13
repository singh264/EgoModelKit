# EgoModelKit

EgoModelKit packages egocentric-video research models behind one command-line interface and one local browser GUI.

Supported model adapters:

<details>
<summary><strong>hand-object-contact</strong></summary>

Detects hands, objects, and hand-object contact in images.

Inputs:

- one `.jpg`, `.jpeg`, `.png`, `.bmp`, or `.webp` image, or
- one directory containing supported images.

Directory input is processed non-recursively.

</details>

<details>
<summary><strong>adl-recognition</strong></summary>

Runs the packaged EgoVizML-based ADL pipeline and generates activity-recognition outputs plus Bandini-style hand-use metrics.

Inputs:

- one `.mp4` video,
- one directory containing `.mp4` videos, or
- an existing `all_preds.pkl` file for CLI prediction reuse.

Directory input is processed non-recursively. Multiple videos selected together in the GUI are treated as one ADL session.

</details>

## Setup

Real model runs currently require Python 3.10 or newer, a Linux environment with an NVIDIA GPU, Docker, and NVIDIA GPU container support.

```bash
git clone https://github.com/singh264/EgoModelKit.git
cd EgoModelKit

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e ".[gui]"
```

The `.[gui]` install includes the CLI and the optional local browser GUI dependencies.

## Dry Run Command

Use `--dry-run` to validate a CLI request without running model inference.

<details>
<summary><strong>hand-object-contact</strong></summary>

```bash
egomodelkit run hand-object-contact \
  --input /path/to/image.jpg \
  --output /path/to/results \
  --dry-run
```

The same dry-run flow also accepts a directory containing one or more supported image files.

</details>

<details>
<summary><strong>adl-recognition</strong></summary>

```bash
egomodelkit run adl-recognition \
  --input /path/to/video.mp4 \
  --output /path/to/results \
  --dry-run
```

The same dry-run flow also accepts a directory containing one or more MP4 videos, or an existing `all_preds.pkl` file.

</details>

## Run Command

Real runs check the required host runtime, prepare or reuse the packaged Docker runtime, and print progress in the terminal.

<details>
<summary><strong>hand-object-contact</strong></summary>

Run one image:

```bash
egomodelkit run hand-object-contact \
  --input /path/to/image.jpg \
  --output /path/to/results
```

Run multiple images from one directory:

```bash
egomodelkit run hand-object-contact \
  --input /path/to/image-directory \
  --output /path/to/results
```

</details>

<details>
<summary><strong>adl-recognition</strong></summary>

Run one video:

```bash
egomodelkit run adl-recognition \
  --input /path/to/video.mp4 \
  --output /path/to/results
```

Run multiple videos from one directory:

```bash
egomodelkit run adl-recognition \
  --input /path/to/video-directory \
  --output /path/to/results
```

Reuse an existing combined prediction file:

```bash
egomodelkit run adl-recognition \
  --input /path/to/all_preds.pkl \
  --output /path/to/results
```

</details>

## Model Outputs

The GUI creates one `run-*` folder inside the selected output folder for each run.

<details>
<summary><strong>Hand-object contact (HOC)</strong></summary>

- <details>
  <summary><strong>Input: one image</strong></summary>

  ```text
  output-root/
    run-YYYY-MM-DD-HHMMSS/
      README.txt
      run_summary.json
      run_manifest.json
      visual_outputs/
        hand_object_contact/
          image_det.png
      technical/
        model_outputs/
          image_shan.json
          image_shan.pkl
      logs/
        progress.jsonl
        runtime.log
  ```

  Review `visual_outputs/hand_object_contact/` first. It contains the annotated detection image.

  </details>

- <details>
  <summary><strong>Input: multiple images</strong></summary>

  ```text
  output-root/
    run-YYYY-MM-DD-HHMMSS/
      README.txt
      run_summary.json
      run_manifest.json
      visual_outputs/
        hand_object_contact/
          image1_det.png
          image2_det.png
          image3_det.png
      technical/
        model_outputs/
          image1_shan.json
          image1_shan.pkl
          image2_shan.json
          image2_shan.pkl
          image3_shan.json
          image3_shan.pkl
      logs/
        progress.jsonl
        runtime.log
  ```

  Review `visual_outputs/hand_object_contact/` first. Each image has a visualization and matching technical model outputs.

  </details>

</details>

<details>
<summary><strong>Activity recognition (ADL)</strong></summary>

- <details>
  <summary><strong>Input: one video</strong></summary>

  ```text
  output-root/
    run-YYYY-MM-DD-HHMMSS/
      README.txt
      run_summary.json
      run_manifest.json
      results/
        video_level_metrics.csv
        session_level_metrics.csv
        video_level_metrics_summary.csv
      technical/
        model_outputs/
          predictions.csv
          predictions_summary.csv
          adl_input_manifest.csv
          all_preds.pkl
        post_processing/
          adl_subclip_manifest.csv
          frame_level_predictions.csv
          interaction_segments.csv
          metrics_config.json
        intermediate_files/
          extracted_frames/
          detic_outputs/
          shan_outputs/
      logs/
        progress.jsonl
        runtime.log
  ```

  Review `results/video_level_metrics.csv` first for the calculated hand-use metrics.

  </details>

- <details>
  <summary><strong>Input: multiple videos</strong></summary>

  ```text
  output-root/
    run-YYYY-MM-DD-HHMMSS/
      README.txt
      run_summary.json
      run_manifest.json
      results/
        video_level_metrics.csv
        session_level_metrics.csv
        video_level_metrics_summary.csv
      technical/
        model_outputs/
          predictions.csv
          predictions_summary.csv
          adl_input_manifest.csv
          all_preds.pkl
        post_processing/
          adl_subclip_manifest.csv
          frame_level_predictions.csv
          interaction_segments.csv
          metrics_config.json
        intermediate_files/
          extracted_frames/
            video1/
            video2/
            video3/
          detic_outputs/
          shan_outputs/
      logs/
        progress.jsonl
        runtime.log
  ```

  The videos are treated as one ADL session. Review `results/session_level_metrics.csv` for the combined session and `results/video_level_metrics.csv` for each video.

  </details>

</details>

<details>
<summary><strong>Output file guide</strong></summary>

| File or folder | Purpose |
|---|---|
| `README.txt` | Plain-language guide for one run folder. |
| `run_summary.json` | Summary of the run and its status. |
| `run_manifest.json` | Record of the run settings and model/runtime versions. |
| `*_det.png` | HOC visualization with model detections. |
| `*_shan.json` | Structured HOC model output. |
| `*_shan.pkl` | Raw HOC model output. |
| `video_level_metrics.csv` | Hand-use metrics for each video. |
| `session_level_metrics.csv` | Combined hand-use metrics for the ADL session. |
| `video_level_metrics_summary.csv` | Compact video-level metric summary. |
| `predictions.csv` | Detailed ADL model predictions. |
| `predictions_summary.csv` | Compact ADL prediction summary. |
| `adl_input_manifest.csv` | Input order and source-video information. |
| `all_preds.pkl` | Combined raw ADL prediction file. |
| `adl_subclip_manifest.csv` | Source-time information used to exclude padded tail frames from metrics. |
| `frame_level_predictions.csv` | Frame-level data used to calculate hand-use metrics. |
| `interaction_segments.csv` | Continuous hand-interaction segments. |
| `metrics_config.json` | Metric and preprocessing settings used for the run. |
| `extracted_frames/` | Video frames used by the ADL pipeline. |
| `detic_outputs/` | Technical Detic outputs. |
| `shan_outputs/` | Technical HOC outputs generated from video frames. |
| `progress.jsonl` | Run progress history. |
| `runtime.log` | Runtime log for troubleshooting. |

</details>

## Local Browser GUI

### Backend Layer

Launch the local GUI and backend:

```bash
egomodelkit gui
```

Useful alternatives:

```bash
egomodelkit gui --no-browser
egomodelkit gui --port 7860 --no-browser
```

The server binds to `127.0.0.1` by default.

### API Layer

<details>
<summary><strong>Local React frontend API</strong></summary>

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/models` | List the models and input types available in the GUI. |
| `POST` | `/api/output-preview` | Build the output-folder preview shown before a run. |
| `POST` | `/api/dry-run` | Check the selected inputs, output folder, and model requirements without running inference. |
| `POST` | `/api/runs` | Check the request and start a model run. |
| `GET` | `/api/runs/{run_id}/progress` | Return the current run status and progress. |
| `POST` | `/api/cancel-run` | Cancel an active dry run or model run. |
| `POST` | `/api/open-output-folder` | Open the output folder for a GUI run. |
| `POST` | `/api/select-output-folder` | Open the system folder picker when available. |

ADL runs can use either the left or right hand as the dominant hand; right is the default.

Interactive FastAPI documentation is also available while the backend is running:

```text
http://127.0.0.1:7860/docs
```

</details>

### Frontend Layer

The React frontend source lives in `src/egomodelkit/web`. Production assets are served from `src/egomodelkit/web/dist`; development uses the Vite server at `127.0.0.1:5173` with `/api` proxied to the local backend.

<details>
<summary><strong>Current GUI features</strong></summary>

- Backend-loaded model selection and model-specific file filtering.
- Single-file and multi-file input selection.
- ADL dominant-hand selection and one-session treatment for multi-video input.
- Output-folder selection, review, dry run, and model execution.
- Consistent runtime preflight before GUI dry runs and real runs.
- Cancellation for active dry runs and model runs.
- Progress polling, refresh persistence, and guarded navigation during active work.
- Completed/failed result views, output-folder preview, and open-output-folder actions.

</details>

## Current Platform Support

<details>
<summary><strong>View platform support</strong></summary>

| Environment | Model runs | Notes |
|---|---|---|
| Linux + NVIDIA GPU | Supported | Primary runtime path. The packaged models and GPU containers are built and tested around Linux and NVIDIA CUDA. |
| Windows + WSL2 + NVIDIA GPU | Supported | Uses the same Linux-based runtime through WSL2. Docker GPU access must be working inside WSL2. |
| Native Windows | Not currently supported | EgoModelKit's Docker-based model runtimes are currently supported on Windows only through WSL2, and the underlying model repositories assume a Linux-style environment. Native support would require a separate Windows runtime and testing path. |
| macOS | Not currently supported for model runs | Suitable for code, documentation, and GUI development, but the packaged models require NVIDIA CUDA GPU support. |
| Linux without an NVIDIA GPU | Not currently supported for model runs | The application can still be used for development and non-model tasks, but model execution requires access to an NVIDIA CUDA GPU. |

`--dry-run` checks a CLI request without running the model. GUI **Dry Run** also checks whether the selected model can run on the current computer.

</details>

## Development

Use Python 3.10 or newer. Run backend/package checks from the repository root before starting the local backend:

```bash
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e ".[dev,gui]"

ruff check .

pytest \
  --cov=egomodelkit \
  --cov-report=term-missing \
  --cov-fail-under=90

python -m build
egomodelkit --help
```

Start the backend in one terminal:

```bash
egomodelkit gui --no-browser
```

For frontend development, use Node.js 22.12.0. In a second terminal, install dependencies, run checks, build, then start Vite:

```bash
cd src/egomodelkit/web
npm ci
npm run typecheck
npm test
npm run build
npm run dev
```
