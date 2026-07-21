import json
import sys
from pathlib import Path
from qubo_project.model import train, predict

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


if __name__ == "__main__":
    demo.launch()
