""" ADL prediction wrapper around EgoVizML's model utilities. """

import argparse
from pathlib import Path

import polars as pl
from egoviz.models import inference, processing

DEFAULT_MODEL_PATH = "models/binary_active_logreg.joblib"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-preds-input", required = True)
    parser.add_argument("--model", default = DEFAULT_MODEL_PATH)
    parser.add_argument("--output", required = True)
    parser.add_argument("--summary-output", required = True)
    
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
    
    results = inference.predict(pl.from_pandas(scaled), model).to_pandas()
    
    probability_columns = [
        column
        for column in results.columns
        if column.startswith("prob_")
    ]
    
    if probability_columns:
        results["predicted_label_readable"] = (
            results[probability_columns]
            .idxmax(axis = 1)
            .str
            .replace("prob_", "", regex = False)
        )
        results["predicted_probability"] = results[probability_columns].max(axis = 1)
    
    for column in results.columns:
        if results[column].dtype == "object":
            results[column] = results[column].map(_safe_csv_value)
    
    output_path = Path(args.output)
    summary_path = Path(args.summary_output)
    
    output_path.parent.mkdir(parents = True, exist_ok = True)
    summary_path.parent.mkdir(parents = True, exist_ok = True)
    
    results.to_csv(output_path, index = False)
    
    summary_columns = [
        "video",
        "adl",
        "predicted_label_readable",
        "predicted_probability",
        "predicted_class",
    ]
    
    summary_columns.extend(
        column
        for column in results.columns
        if column.startswith("prob_")
    )
    
    results[
        [
            column
            for column in summary_columns
            if column in results.columns
        ]
    ].to_csv(
        summary_path,
        index = False
    )
    
    print(f"Saved predictions to: {output_path}", flush = True)
    print(f"Saved prediction summary to: {summary_path}", flush = True)
    
def _safe_csv_value(value: object) -> str:
    if isinstance(value, bytes):
        return repr(value)
    
    return str(value)

if __name__ == "__main__":
    main()
