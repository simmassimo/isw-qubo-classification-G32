import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qubo_project.model import train, predict
from qubo_project.preprocessing import fit_normalize
import os

import gradio as gr

OUTPUTS_DIR = Path(__file__).resolve().parents[2] / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)


def run_train(csv_file, target_column, classifier, seed):
    if csv_file is None:
        raise gr.Error("Please upload a training CSV file.")
    if not target_column:
        raise gr.Error("Please provide the target column name.")

    model_path = str(OUTPUTS_DIR / "model.joblib")
    metrics_path = str(OUTPUTS_DIR / "train_metrics.json")

    train(
        classifier=classifier,
        reducedTrain_csv=csv_file,
        target_column=target_column,
        model_path=model_path,
        metrics_json=metrics_path,
        seed=int(seed),
    )

    with open(metrics_path) as f:
        metrics = json.load(f)

    return metrics, model_path


def run_predict(csv_file, target_column, model_file):
    if csv_file is None:
        raise gr.Error("Please upload a test CSV file.")
    if model_file is None:
        raise gr.Error("Please upload a trained model file (.joblib).")
    if not target_column:
        raise gr.Error("Please provide the target column name.")

    predictions_path = str(OUTPUTS_DIR / "predictions.csv")
    stats_path = str(OUTPUTS_DIR / "predict_stats.json")

    predict(
        reduced_Test_csv=csv_file,
        target_column=target_column,
        model_path=model_file,
        predictions_csv=predictions_path,
        classif_stats_json=stats_path,
    )

    with open(stats_path) as f:
        stats = json.load(f)

    return stats, predictions_path

def run_fit_normalize(
    input_csv: str,
    target_column: str,
    normalized_csv: str,  # Name of output normalized data set
    outInitalRes_json: str,  # Name of output statistics and data file
    minPercValid: float = 0.05,
):
    running_update = gr.update(value="⏳ Running...", interactive=False)
    idle_update = gr.update(value="Fit & Normalize", interactive=True)

    yield gr.update(), gr.update(), running_update

    try:
        if input_csv is None:
            raise gr.Error("Please upload a CSV file.")
        if not target_column:
            raise gr.Error("Please provide the target column name.")
        if not normalized_csv:
            raise gr.Error("Please provide a filename for the normalized CSV.")
        if not outInitalRes_json:
            raise gr.Error("Please provide a filename for the fit & normalize statistics JSON.")

        preprocessed_path = str(OUTPUTS_DIR / normalized_csv)
        preprocessed_stats_path = str(OUTPUTS_DIR / outInitalRes_json)

        try:
            p, t, s = fit_normalize(
                input_csv=input_csv,
                target_column=target_column,
                normalized_csv=preprocessed_path,
                outInitalRes_json=preprocessed_stats_path,
                minPercValid=minPercValid,
            )
        except gr.Error:
            raise
        except Exception as e:
            raise gr.Error(f"Error during fit & normalize: {e}")

        yield s, preprocessed_path, idle_update
    except gr.Error:
        yield gr.update(), gr.update(), idle_update
        raise


with gr.Blocks(title="QUBO Classification") as demo:
    gr.Markdown("# QUBO Classification")

    with gr.Tab("Train"):
        train_csv = gr.File(label="Training CSV")
        train_target = gr.Textbox(label="Target column", value="target")
        train_classifier = gr.Dropdown(
            choices=["random_forest"], value="random_forest", label="Classifier"
        )
        train_seed = gr.Number(label="Seed", value=42, precision=0)
        train_button = gr.Button("Train")
        train_metrics_output = gr.JSON(label="Training metrics")
        train_model_output = gr.File(label="Trained model")

        train_button.click(
            fn=run_train,
            inputs=[train_csv, train_target, train_classifier, train_seed],
            outputs=[train_metrics_output, train_model_output],
        )

    with gr.Tab("Predict"):
        predict_csv = gr.File(label="Test CSV")
        predict_target = gr.Textbox(label="Target column", value="target")
        predict_model = gr.File(label="Trained model (.joblib)")
        predict_button = gr.Button("Predict")
        predict_stats_output = gr.JSON(label="Classification statistics")
        predict_csv_output = gr.File(label="Predictions CSV")

        predict_button.click(
            fn=run_predict,
            inputs=[predict_csv, predict_target, predict_model],
            outputs=[predict_stats_output, predict_csv_output],
        )

    with gr.Tab("Fit & Normalize"):
        fit_normalize_csv = gr.File(label="CSV to fit & normalize", file_types=[".csv"])
        fit_normalize_target = gr.Textbox(label="Target column", value="target")
        with gr.Row():
            fit_normalize_output_csv = gr.Textbox(label="Normalized CSV filename", value="normalized_data.csv")
            fit_normalize_output_json = gr.Textbox(label="Fit & Normalize statistics filename", value="fit_normalize_stats.json")
        with gr.Accordion("Advanced", open=False):
            gr.Markdown("Adjust the minimum percentage of valid non-zero data for a column. Columns with less than this percentage will be removed during preprocessing.")
            fit_normalize_min_perc_valid = gr.Slider(label="Minimum % of valid non-zero data for a column", minimum=0.0, maximum=1.0, value=0.05, step=0.01)
        fit_normalize_button = gr.Button("Fit & Normalize")
        with gr.Accordion("# Output", open=False) as fit_normalize_output_accordion:
            gr.Markdown("The output will include the normalized CSV file and a JSON file containing statistics about the fit & normalize process.")
            fit_normalize_stats_output = gr.JSON(label="Fit & Normalize statistics")
            fit_normalize_csv_output = gr.File(label="Normalized CSV")

        fit_normalize_button.click(
            fn=run_fit_normalize,
            inputs=[fit_normalize_csv, fit_normalize_target, fit_normalize_output_csv, fit_normalize_output_json, fit_normalize_min_perc_valid],
            outputs=[fit_normalize_stats_output, fit_normalize_csv_output, fit_normalize_button],
        ).success( lambda: fit_normalize_output_accordion.update(visible=True) )


if __name__ == "__main__":
    demo.launch()
