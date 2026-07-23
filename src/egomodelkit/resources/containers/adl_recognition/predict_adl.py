"""ADL prediction wrapper around EgoVizML's model utilities."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Final

import pandas as pd
import polars as pl
from egoviz.models import inference, processing

DEFAULT_MODEL_PATH: Final[str] = "models/binary_active_logreg.joblib"
DEFAULT_SESSION_ID: Final[str] = "session001"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-preds-input", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--segment-output", required=True)
    parser.add_argument("--video-summary-output", required=True)
    parser.add_argument("--session-summary-output", required=True)
    parser.add_argument("--segment-manifest")
    parser.add_argument("--input-manifest")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    preds = processing.load_pickle(args.all_preds_input)

    frame_df = processing.generate_df_from_preds(preds)
    features = processing.generate_binary_presence_df(frame_df)
    scaled = processing.row_wise_min_max_scaling(features)
    scaled = scaled.fillna(0.0)

    model = inference.load_production_model(args.model)

    for column in model.feature_names:
        if column not in scaled.columns:
            scaled[column] = 0.0

    raw_results = inference.predict(pl.from_pandas(scaled), model).to_pandas()
    segment_predictions = build_segment_predictions(
        raw_results,
        segment_manifest_path=_optional_path(args.segment_manifest),
    )
    video_summary = build_video_summary(
        segment_predictions,
        input_manifest_path=_optional_path(args.input_manifest),
    )
    session_summary = build_session_summary(segment_predictions)

    outputs = (
        (Path(args.segment_output), segment_predictions),
        (Path(args.video_summary_output), video_summary),
        (Path(args.session_summary_output), session_summary),
    )

    for output_path, dataframe in outputs:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(output_path, index=False)

    print(f"Saved segment predictions to: {args.segment_output}", flush=True)
    print(f"Saved video summary to: {args.video_summary_output}", flush=True)
    print(f"Saved session summary to: {args.session_summary_output}", flush=True)


def build_segment_predictions(
    results: pd.DataFrame,
    *,
    segment_manifest_path: Path | None,
) -> pd.DataFrame:
    """Join classifier outputs to source-video segment metadata."""
    normalized = results.copy()

    if "video" not in normalized.columns:
        raise RuntimeError("EgoVizML predictions are missing the required 'video' column.")

    normalized["segment_name"] = normalized["video"].map(_normalized_text)

    probability_columns = sorted(
        column for column in normalized.columns if column.startswith("prob_")
    )

    if probability_columns:
        normalized["predicted_probability"] = normalized[probability_columns].max(axis=1)

    normalized["predicted_adl"] = _readable_prediction_labels(
        normalized,
        probability_columns=probability_columns,
    )

    prediction_columns = [
        "segment_name",
        "predicted_adl",
        "predicted_probability",
        "predicted_class",
        *probability_columns,
    ]
    prediction_columns = [
        column for column in prediction_columns if column in normalized.columns
    ]
    normalized = normalized[prediction_columns]

    if normalized["segment_name"].duplicated().any():
        raise RuntimeError("EgoVizML produced duplicate predictions for one or more segments.")

    if segment_manifest_path is not None and segment_manifest_path.is_file():
        segment_manifest = pd.read_csv(segment_manifest_path)
        required_columns = {
            "session_id",
            "source_video",
            "segment_name",
            "segment_index",
            "start_time_seconds",
            "end_time_seconds",
            "valid_duration_seconds",
        }
        missing_columns = required_columns - set(segment_manifest.columns)

        if missing_columns:
            raise RuntimeError(
                "ADL segment manifest is missing required columns: "
                + ", ".join(sorted(missing_columns))
            )

        output = segment_manifest.merge(
            normalized,
            on="segment_name",
            how="left",
            validate="one_to_one",
        )
    else:
        output = normalized.copy()
        output.insert(0, "session_id", DEFAULT_SESSION_ID)
        output.insert(
            1,
            "source_video",
            output["segment_name"].map(_source_video_from_segment_name),
        )
        output.insert(
            3,
            "segment_index",
            output["segment_name"].map(_segment_index_from_name),
        )
        output.insert(4, "start_time_seconds", pd.NA)
        output.insert(5, "end_time_seconds", pd.NA)
        output.insert(6, "valid_duration_seconds", pd.NA)

    output["prediction_status"] = output["predicted_adl"].apply(
        lambda value: "predicted" if pd.notna(value) and str(value) else "missing"
    )

    preferred_columns = [
        "session_id",
        "source_video",
        "segment_name",
        "segment_index",
        "start_time_seconds",
        "end_time_seconds",
        "valid_duration_seconds",
        "prediction_status",
        "predicted_adl",
        "predicted_probability",
        "predicted_class",
        *probability_columns,
    ]

    return output[
        [column for column in preferred_columns if column in output.columns]
    ].sort_values(
        ["session_id", "source_video", "segment_index"],
        kind="stable",
        ignore_index=True,
    )


def build_video_summary(
    segment_predictions: pd.DataFrame,
    *,
    input_manifest_path: Path | None,
) -> pd.DataFrame:
    """Build descriptive per-video ADL prediction counts and durations."""
    duration_by_video: dict[str, float] = {}

    if input_manifest_path is not None and input_manifest_path.is_file():
        input_manifest = pd.read_csv(input_manifest_path)

        if {"input_name", "source_duration_seconds"}.issubset(input_manifest.columns):
            duration_by_video = {
                _normalized_text(row["input_name"]): float(row["source_duration_seconds"])
                for _, row in input_manifest.iterrows()
            }

    rows: list[dict[str, object]] = []

    for (session_id, source_video), group in segment_predictions.groupby(
        ["session_id", "source_video"],
        sort=False,
        dropna=False,
    ):
        predicted = group[group["prediction_status"] == "predicted"]
        valid_durations = pd.to_numeric(group["valid_duration_seconds"], errors="coerce")
        source_video_text = _normalized_text(source_video)
        source_duration = duration_by_video.get(source_video_text)

        if source_duration is None:
            source_duration = float(valid_durations.fillna(0.0).sum())

        rows.append(
            {
                "session_id": _normalized_text(session_id),
                "source_video": source_video_text,
                "source_duration_seconds": source_duration,
                "segment_count": len(group),
                "predicted_segment_count": len(predicted),
                "total_valid_duration_seconds": float(valid_durations.fillna(0.0).sum()),
                "predicted_adl_counts": _label_counts_json(predicted),
                "predicted_adl_valid_duration_seconds": _label_durations_json(predicted),
            }
        )

    return pd.DataFrame(rows)


def build_session_summary(segment_predictions: pd.DataFrame) -> pd.DataFrame:
    """Build descriptive per-session ADL prediction counts and durations."""
    rows: list[dict[str, object]] = []

    for session_id, group in segment_predictions.groupby(
        "session_id",
        sort=False,
        dropna=False,
    ):
        predicted = group[group["prediction_status"] == "predicted"]
        valid_durations = pd.to_numeric(group["valid_duration_seconds"], errors="coerce")

        rows.append(
            {
                "session_id": _normalized_text(session_id),
                "input_video_count": group["source_video"].nunique(dropna=True),
                "segment_count": len(group),
                "predicted_segment_count": len(predicted),
                "total_valid_duration_seconds": float(valid_durations.fillna(0.0).sum()),
                "predicted_adl_counts": _label_counts_json(predicted),
                "predicted_adl_valid_duration_seconds": _label_durations_json(predicted),
            }
        )

    return pd.DataFrame(rows)


def _label_counts_json(predictions: pd.DataFrame) -> str:
    counts = predictions["predicted_adl"].value_counts().sort_index()
    return json.dumps({str(label): int(count) for label, count in counts.items()}, sort_keys=True)


def _label_durations_json(predictions: pd.DataFrame) -> str:
    if predictions.empty:
        return "{}"

    durations = pd.to_numeric(
        predictions["valid_duration_seconds"],
        errors="coerce",
    ).fillna(0.0)
    grouped = durations.groupby(predictions["predicted_adl"]).sum().sort_index()
    return json.dumps(
        {str(label): float(duration) for label, duration in grouped.items()},
        sort_keys=True,
    )


def _source_video_from_segment_name(segment_name: object) -> str:
    return re.sub(r"--\d+$", "", _normalized_text(segment_name))


def _segment_index_from_name(segment_name: object) -> int:
    match = re.search(r"--(\d+)$", _normalized_text(segment_name))
    return int(match.group(1)) if match else 1


def _readable_prediction_labels(
    results: pd.DataFrame,
    *,
    probability_columns: list[str],
) -> pd.Series:
    """Return stable human-readable labels without changing model decisions."""
    if probability_columns:
        maximum_probability_columns = results[probability_columns].idxmax(axis=1)
        return maximum_probability_columns.map(
            lambda column: (
                column.removeprefix("prob_") if isinstance(column, str) else ""
            )
        )

    if "predicted_label" not in results.columns:
        return pd.Series("", index=results.index, dtype="object")

    return results["predicted_label"].map(_safe_prediction_label)


def _safe_prediction_label(value: object) -> str:
    """Normalize a fallback label and reject binary-corrupted text."""
    text = _normalized_text(value).strip()

    if "\x00" in text or "\ufffd" in text:
        return ""

    return text


def _normalized_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    return str(value)


def _optional_path(value: str | None) -> Path | None:
    return Path(value) if value else None


if __name__ == "__main__":
    main()
