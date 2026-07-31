"""Combine Detic and hand-object predictions with per-frame progress."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Final

import numpy as np
import torch
from egoviz.models.processing import load_pickle
from torchvision.ops import box_iou

EGOVIZML_HOME: Final[Path] = Path("/opt/EgoVizML")
PROGRESS_PREFIX: Final[str] = "EGOMODELKIT_PROGRESS "
ADL_NAMES: Final[tuple[str, ...]] = (
    "communication-management",
    "functional-mobility",
    "grooming-health-management",
    "home-management",
    "leisure-other-activities",
    "meal-preparation-cleanup",
    "self-feeding",
)

sys.path.insert(0, str(EGOVIZML_HOME / "scripts"))
from process_detic import _load_mapping_df, process_detic_preds  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine Detic and hand-object predictions."
    )
    parser.add_argument("data_root")
    parser.add_argument("--active-iou", type=float, default=0.75)
    parser.add_argument("--progress-offset", type=int, default=0)
    parser.add_argument("--progress-total", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    combine_predictions(
        data_root=Path(args.data_root),
        active_iou=args.active_iou,
        progress_offset=args.progress_offset,
        progress_total=args.progress_total,
    )


def combine_predictions(
    *,
    data_root: Path,
    active_iou: float,
    progress_offset: int,
    progress_total: int,
) -> dict[str, dict[str, object]]:
    pairs = _prediction_pairs(data_root)
    if not pairs:
        raise RuntimeError("No paired Detic and hand-object predictions were found.")

    expected_total = progress_offset + len(pairs)
    if progress_total != expected_total:
        raise RuntimeError(
            "ADL prediction progress total does not match the discovered frame count: "
            f"expected {expected_total}, received {progress_total}."
        )

    mapping_df = _load_mapping_df()
    all_preds: dict[str, dict[str, object]] = {}

    for processed, (adl_name, detic_path, shan_path) in enumerate(pairs, start=1):
        detic_preds = process_detic_preds(load_pickle(str(detic_path)), mapping_df)
        shan_preds = load_pickle(str(shan_path))
        shan_boxes = shan_preds["objects"] if shan_preds is not None else None
        detic_preds["active_objects"] = get_active_objects(
            detic_preds["boxes"],
            shan_boxes,
            active_iou=active_iou,
        )

        filename_parts = detic_path.name.split("_")
        if len(filename_parts) < 2:
            raise RuntimeError(f"Unexpected Detic prediction filename: {detic_path.name}")
        video = f"{filename_parts[0]}_{filename_parts[1]}"
        all_preds[f"{adl_name}_{video}"] = detic_preds

        _emit_progress(
            "adl_prediction_frame_processed",
            current=progress_offset + processed,
            total=progress_total,
        )

    with (data_root / "all_preds.pkl").open("wb") as stream:
        pickle.dump(all_preds, stream)

    return all_preds


def _prediction_pairs(data_root: Path) -> list[tuple[str, Path, Path]]:
    pairs: list[tuple[str, Path, Path]] = []

    for adl_name in ADL_NAMES:
        detic_dir = data_root / adl_name / "detic"
        shan_dir = data_root / adl_name / "shan"
        detic_by_base = {
            path.name.removesuffix("_detic.pkl"): path
            for path in sorted(detic_dir.glob("*_detic.pkl"))
        }
        shan_by_base = {
            path.name.removesuffix("_shan.pkl"): path
            for path in sorted(shan_dir.glob("*_shan.pkl"))
        }

        missing_shan = sorted(set(detic_by_base) - set(shan_by_base))
        missing_detic = sorted(set(shan_by_base) - set(detic_by_base))
        if missing_shan or missing_detic:
            details: list[str] = []
            if missing_shan:
                details.append("missing Shan: " + ", ".join(missing_shan[:5]))
            if missing_detic:
                details.append("missing Detic: " + ", ".join(missing_detic[:5]))
            raise RuntimeError(
                f"Mismatched ADL prediction files for {adl_name}: " + "; ".join(details)
            )

        pairs.extend(
            (adl_name, detic_by_base[base], shan_by_base[base])
            for base in sorted(detic_by_base)
        )

    return pairs


def get_active_objects(
    detic_boxes: list[object] | np.ndarray,
    shan_boxes: list[object] | np.ndarray | None,
    *,
    active_iou: float,
) -> list[bool] | np.ndarray:
    """Return whether each Detic object overlaps a contacted Shan object."""
    if shan_boxes is not None:
        if len(detic_boxes) == 0:
            return np.array([])

        normalized_detic_boxes = np.asarray(detic_boxes).astype(int)
        normalized_shan_boxes = np.asarray([obj[0:4] for obj in shan_boxes]).astype(int)
        ious = box_iou(
            torch.tensor(normalized_detic_boxes),
            torch.tensor(normalized_shan_boxes),
        )
        return [bool(torch.any(ious[index] >= active_iou)) for index in range(len(ious))]

    if len(detic_boxes) > 0:
        return np.array([False] * len(detic_boxes))

    return np.array([])


def _emit_progress(kind: str, **payload: object) -> None:
    print(
        PROGRESS_PREFIX + json.dumps({"kind": kind, **payload}, sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
