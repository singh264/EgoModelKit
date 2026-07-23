# EgoModelKit

EgoModelKit packages egocentric-video research models behind one command-line interface and one local browser GUI.

Supported model adapters:

<details>
<summary><strong>hand-interaction</strong></summary>

Measures functional hand-object interactions in egocentric videos using hand-object-contact detections and hand-use processing [1]. Outputs include frame profiles, segments, and Perc, Dur, and Num.

Inputs:

- one `.mp4` video, or
- one directory containing `.mp4` videos.

Directory input is non-recursive. Multiple videos form one session: Statepool resets per video, session segments may merge across adjacent videos, and per-video metrics remain separate. Processing uses 30 FPS, 720 × 405 frames, 30-frame Statepool, and contact state ≥3. The dominant hand defaults to right.

</details>

<details>
<summary><strong>adl-recognition</strong></summary>

Runs the packaged EgoVizML activity-recognition pipeline [2], [3]. Each original video is divided independently into segments of at most 60 seconds, and inference frames are sampled at 1 FPS. Detic object detections [4], [5] and hand-object-contact detections [6], [7] are retained to generate active-object features. The Detic model relies on Detectron2 [8]. An object is active when the Detic/100DOH bounding-box IoU is greater than 0.8. Outputs are ADL segment predictions and descriptive video/session summaries.

Inputs:

- one `.mp4` video,
- one directory containing `.mp4` videos, or
- an existing `all_preds.pkl` file for CLI prediction reuse.

Directory input is processed non-recursively. Multiple videos provided together through either the CLI or GUI are treated as one ADL session.

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
<summary><strong>hand-interaction</strong></summary>

```bash
egomodelkit run hand-interaction \
  --input /path/to/video.mp4 \
  --output /path/to/results \
  --dominant-hand right \
  --dry-run
```

Directory input and `--dominant-hand left` are also supported.

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
<summary><strong>hand-interaction</strong></summary>

Run one video:

```bash
egomodelkit run hand-interaction \
  --input /path/to/video.mp4 \
  --output /path/to/results \
  --dominant-hand right
```

Run multiple videos as one session:

```bash
egomodelkit run hand-interaction \
  --input /path/to/video-directory \
  --output /path/to/results \
  --dominant-hand left
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

Both the CLI and GUI create one `run-*` folder inside the selected output folder for each run.

<details>
<summary><strong>Hand interaction</strong></summary>

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
          hand_interaction_input_manifest.csv
        post_processing/
          hand_interaction_subclip_manifest.csv
          frame_level_predictions.csv
          interaction_segments.csv
          metrics_config.json
        intermediate_files/
          extracted_frames/
            video001--1/
            ...
          shan_outputs/
            video001--1/
            ...
      logs/
        progress.jsonl
        runtime.log
  ```

  Review `results/video_level_metrics.csv` first.

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
          hand_interaction_input_manifest.csv
        post_processing/
          hand_interaction_subclip_manifest.csv
          frame_level_predictions.csv
          interaction_segments.csv
          metrics_config.json
        intermediate_files/
          extracted_frames/
            video001--1/
            ...
            video002--1/
            ...
          shan_outputs/
            video001--1/
            ...
            video002--1/
            ...
      logs/
        progress.jsonl
        runtime.log
  ```

  Review `results/session_level_metrics.csv` for the combined session and `results/video_level_metrics.csv` for each video.

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
        adl_segment_predictions.csv
        adl_video_summary.csv
        adl_session_summary.csv
      technical/
        model_outputs/
          adl_input_manifest.csv
          all_preds.pkl
        post_processing/
          adl_segment_manifest.csv
          adl_processing_config.json
        intermediate_files/
          extracted_frames/
          detic_outputs/
          shan_outputs/
      logs/
        progress.jsonl
        runtime.log
  ```

  Review `results/adl_segment_predictions.csv` first. The video and session files contain descriptive prediction summaries only.

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
        adl_segment_predictions.csv
        adl_video_summary.csv
        adl_session_summary.csv
      technical/
        model_outputs/
          adl_input_manifest.csv
          all_preds.pkl
        post_processing/
          adl_segment_manifest.csv
          adl_processing_config.json
        intermediate_files/
          extracted_frames/
            video001--1/
            ...
            video002--1/
            ...
          detic_outputs/
          shan_outputs/
      logs/
        progress.jsonl
        runtime.log
  ```

  Review `results/adl_segment_predictions.csv` for segment labels, `results/adl_video_summary.csv` for per-video totals, and `results/adl_session_summary.csv` for the combined descriptive summary.

  </details>

</details>

<details>
<summary><strong>Output file guide</strong></summary>

| File or folder | Purpose |
|---|---|
| `README.txt` | Plain-language guide for one run folder. |
| `run_summary.json` | Summary of the run and its status. |
| `run_manifest.json` | Record of the run settings and model/runtime versions. |
| `video_level_metrics.csv` | Hand-use metrics for each video. |
| `session_level_metrics.csv` | Combined hand-use metrics for the input-video session. |
| `video_level_metrics_summary.csv` | Compact video-level metric summary. |
| `adl_segment_predictions.csv` | Per-segment ADL labels, confidence, probabilities, and source-time metadata. |
| `adl_video_summary.csv` | Descriptive predicted-ADL counts and valid durations for each original video. |
| `adl_session_summary.csv` | Descriptive predicted-ADL counts and valid durations across the selected session. |
| `adl_input_manifest.csv` | Input order and source-video information. |
| `hand_interaction_input_manifest.csv` | Hand-interaction input order and source-video information. |
| `all_preds.pkl` | Combined raw ADL prediction file. |
| `adl_segment_manifest.csv` | Source-video, segment index, start/end time, and final partial-segment duration. |
| `adl_processing_config.json` | ADL segmentation, sampling, active-object, and feature-processing settings. |
| `hand_interaction_subclip_manifest.csv` | Hand-interaction source-time and analyzed-frame mapping. |
| `frame_level_predictions.csv` | Frame-level data used to calculate hand-use metrics. |
| `interaction_segments.csv` | Continuous hand-interaction segments. |
| `metrics_config.json` | Metric and preprocessing settings used for the run. |
| `extracted_frames/` | Video frames used by the video pipeline. |
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

Hand-interaction runs support left- or right-hand dominance; right is the default. ADL recognition does not use a dominant-hand setting.

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
- Dominant-hand selection for hand-interaction and one-session treatment for multi-video inputs.
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

## References

<details>
<summary><strong>View references</strong></summary>

[1] A. Bandini, M. Dousty, S. L. Hitzig, B. C. Craven, S. Kalsi-Ryan, and J. Zariffa, “Measuring hand use in the home after cervical spinal cord injury using egocentric video,” *Journal of Neurotrauma*, vol. 39, nos. 23–24, pp. 1697–1707, Dec. 2022. [Online]. Available: [Publisher website](https://journals.sagepub.com/doi/10.1089/neu.2022.0156)

[2] A. Kadambi and J. Zariffa, “Detecting activities of daily living in egocentric video to contextualize hand use at home in outpatient neurorehabilitation settings,” *IEEE Transactions on Neural Systems and Rehabilitation Engineering*, vol. 33, pp. 1951–1957, 2025. [Online]. Available: [IEEE Xplore](https://ieeexplore.ieee.org/document/11000436)

[3] A. Kadambi, “EgoVizML,” GitHub repository. [Online]. Available: [https://github.com/adeshkadambi/EgoVizML](https://github.com/adeshkadambi/EgoVizML)

[4] X. Zhou, R. Girdhar, A. Joulin, P. Krähenbühl, and I. Misra, “Detecting twenty-thousand classes using image-level supervision,” in *Computer Vision – ECCV 2022*, Lecture Notes in Computer Science, vol. 13669, pp. 350–368, 2022. [Online]. Available: [Springer Nature](https://link.springer.com/chapter/10.1007/978-3-031-20077-9_21)

[5] Meta AI Research, “Detic,” GitHub repository. [Online]. Available: [https://github.com/facebookresearch/Detic](https://github.com/facebookresearch/Detic)

[6] D. Shan, J. Geng, M. Shu, and D. F. Fouhey, “Understanding human hands in contact at Internet scale,” in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, pp. 9866–9875, Jun. 2020. [Online]. Available: [IEEE Xplore](https://ieeexplore.ieee.org/document/9157473)

[7] D. Shan, J. Geng, M. Shu, and D. F. Fouhey, “Hand Object Detector,” GitHub repository. [Online]. Available: [https://github.com/ddshan/hand_object_detector](https://github.com/ddshan/hand_object_detector)

[8] Y. Wu, A. Kirillov, F. Massa, W.-Y. Lo, and R. Girshick, “Detectron2,” GitHub repository, 2019. [Online]. Available: [https://github.com/facebookresearch/detectron2](https://github.com/facebookresearch/detectron2)

</details>
