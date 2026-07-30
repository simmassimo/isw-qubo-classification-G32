import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qubo_project.model import train, predict
from qubo_project.preprocessing import fit_normalize
from qubo_project.feature_selection import select_features
import os

import gradio as gr

OUTPUTS_DIR = Path(__file__).resolve().parents[2] / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)


def run_train(csv_file, target_column, classifier, seed, model_name="model.joblib", metrics_name="train_metrics.json"):
    running_update = gr.update(value="⏳ Running...", interactive=False)
    idle_update = gr.update(value="Train", interactive=True)

    yield gr.update(), gr.update(), gr.update(), running_update

    try:
        if csv_file is None:
            raise gr.Error("Please upload a training CSV file.")
        if not target_column:
            raise gr.Error("Please provide the target column name.")

        model_path = str(OUTPUTS_DIR / model_name)
        metrics_path = str(OUTPUTS_DIR / metrics_name)

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

        yield metrics, model_path, metrics_path, idle_update
    except gr.Error:
        raise
    except Exception as e:
        yield gr.update(), gr.update(), gr.update(), idle_update
        raise gr.Error(f"Error during Training: {e}")


def run_predict(csv_file, target_column, model_file):
    running_update = gr.update(value="⏳ Running...", interactive=False)
    idle_update = gr.update(value="Predict", interactive=True)

    yield gr.update(), gr.update(), gr.update(), running_update

    try:
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

        yield stats, predictions_path, stats_path, idle_update
    except gr.Error:
        raise
    except Exception as e:
        yield gr.update(), gr.update(), gr.update(), idle_update
        raise gr.Error(f"Error during Prediction: {e}")
    
def run_fit_normalize(
    input_csv: str,
    target_column: str,
    normalized_csv: str,  # Name of output normalized data set
    outInitalRes_json: str,  # Name of output statistics and data file
    minPercValid: float = 0.05,
):
    running_update = gr.update(value="⏳ Running...", interactive=False)
    idle_update = gr.update(value="Fit & Normalize", interactive=True)

    yield gr.update(), gr.update(), gr.update(), running_update

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
            fit_normalize(
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

        with open(preprocessed_stats_path) as f:
            stats = json.load(f)

        yield stats, preprocessed_path, preprocessed_stats_path, idle_update
    except gr.Error:
        yield gr.update(), gr.update(), gr.update(), idle_update
        raise

def run_feature_selection(
    normalized_csv: str, # Input dataset name
    reducedTrain_csv: str, # Name of output training dataset with reduced feat.
    reducedTest_csv: str,  # Name of output test dataset with reduced features
    output_ottim_csv: str, # Name of output optimization data varying alpha
    output_json: str, # Name of output statistics and data file
    target_column: str, # Column name of target
    percTest: float, # % of test data with respect to the dataset size
    percSelected: float, # percentage of features to select
    allowance: int, # Allowance of features to select
    seed: int, # Seed for random repeatibility
    alpha_computations: int, # Max. n. of optimizations varying alpha
    ):
    running_update = gr.update(value="⏳ Running...", interactive=False)
    idle_update = gr.update(value="Feature Selection", interactive=True)

    yield gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), running_update

    try:
        if normalized_csv is None:
            raise gr.Error("Please upload a CSV file.")
        if not target_column:
            raise gr.Error("Please provide the target column name.")
        if not reducedTrain_csv:
            raise gr.Error("Please provide a filename for the reduced training CSV.")
        if not reducedTest_csv:
            raise gr.Error("Please provide a filename for the reduced test CSV.")
        if not output_ottim_csv:
            raise gr.Error("Please provide a filename for the output optimization CSV.")
        if not output_json:
            raise gr.Error("Please provide a filename for the feature selection statistics JSON.")

        reducedTrain_path = str(OUTPUTS_DIR / reducedTrain_csv)
        reducedTest_path = str(OUTPUTS_DIR / reducedTest_csv)
        output_ottim_path = str(OUTPUTS_DIR / output_ottim_csv)
        output_json_path = str(OUTPUTS_DIR / output_json)

        try:
            select_features(
                normalized_csv=normalized_csv,
                target_column=target_column,
                reducedTrain_csv=reducedTrain_path,
                reducedTest_csv=reducedTest_path,
                output_ottim_csv=output_ottim_path,
                output_json=output_json_path,
                percTest=percTest,
                percSelected=percSelected,
                allowance=allowance,
                seed=seed,
                alpha_computations=alpha_computations,
            )
        except gr.Error:
            raise
        except Exception as e:
            raise gr.Error(f"Error during feature selection: {e}")

        with open(output_json_path) as f:
            stats = json.load(f)

        yield stats, reducedTrain_path, reducedTest_path, output_ottim_path, output_json_path, idle_update
    except gr.Error:
        yield gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), idle_update
        raise


with gr.Blocks(title="QUBO Classification") as demo:
    gr.Markdown("# QUBO Classification")

    with gr.Tab("Preprocessing"):
        fn_csv = gr.File(label="CSV to fit & normalize", file_types=[".csv"])
        fn_target = gr.Textbox(label="Target column", value="target")
        with gr.Row():
            fn_output_csv = gr.Textbox(label="Normalized CSV filename", value="normalized_data.csv")
            fn_output_json = gr.Textbox(label="Fit & Normalize statistics filename", value="fit_normalize_stats.json")
        with gr.Accordion("Advanced", open=False):
            gr.Markdown("Adjust the minimum percentage of valid non-zero data for a column. Columns with less than this percentage will be removed during preprocessing.")
            fn_min_perc_valid = gr.Slider(label="Minimum % of valid non-zero data for a column", minimum=0.0, maximum=1.0, value=0.05, step=0.01)

        fn_button = gr.Button("Fit & Normalize")
        gr.Markdown("## Output\nThe output will include the normalized CSV file and a JSON file containing statistics about the fit & normalize process.")
        fn_stats = gr.JSON(label="Fit & Normalize statistics")
        fn_stats_output = gr.File(label="Fit & Normalize statistics", file_types=[".json"])
        fn_csv_output = gr.File(label="Normalized CSV", file_types=[".csv"])

        fn_button.click(
            fn=run_fit_normalize,
            inputs=[fn_csv, fn_target, fn_output_csv, fn_output_json, fn_min_perc_valid],
            outputs=[fn_stats, fn_csv_output, fn_stats_output, fn_button],
        )

    with gr.Tab("Feature Selection"):
            fs_normalized_csv = gr.File(label="CSV to select features from", file_types=[".csv"])
            with gr.Row():
                fs_output_train_csv = gr.Textbox(label="Name of output training dataset with reduced feat.", value="reduced_train_dataset.csv")
                fs_output_test_csv = gr.Textbox(label="Name of output test dataset with reduced features", value="reduced_test_dataset.csv")
            fs_output_optim_csv = gr.Textbox(label="Name of output optimization data varying alpha", value="optimization_data.csv")
            fs_output_json = gr.Textbox(label="Name of output statistics and data file", value="feature_selection_stats.json")
            fs_target = gr.Textbox(label="Target column", value="target")
            with gr.Accordion("Advanced", open=True):
                gr.Markdown("Adjust the minimum percentage of valid non-zero data for a column. Columns with less than this percentage will be removed during preprocessing.")
                fs_perc_test = gr.Slider(label=" % of test data with respect to the dataset size", minimum=0.0, maximum=1.0, value=0.2, step=0.01)
                fs_perc_selected = gr.Slider(label="percentage of features to select", minimum=0.0, maximum=1.0, value=0.2, step=0.01)
            fs_allowance = gr.Number(label="Allowance of features to select", value=0.01, precision=4)
            fs_seed = gr.Number(label="Seed", value=42, precision=0)
            fs_alpha_computations = gr.Number(label="Max. n. of optimizations varying alpha", value=10, precision=0)
            fs_button = gr.Button("Feature Selection")
            with gr.Accordion("Output", open=False) as fs_output_accordion:
                gr.Markdown("The output will include the JSON file containing statistics about the feature selection process.")
                fs_stats_output = gr.JSON(label="Feature Selection statistics")
                fs_csv_train = gr.File(label="Reduced training CSV")
                fs_csv_test = gr.File(label="Reduced test CSV")
                fs_csv_optim = gr.File(label="Optimization data CSV")
                fs_json_output = gr.File(label="Feature Selection statistics JSON")

            fs_button.click(
                fn= run_feature_selection,
                inputs=[fs_normalized_csv, fs_output_train_csv, fs_output_test_csv, fs_output_optim_csv, fs_output_json, fs_target, fs_perc_test, fs_perc_selected, fs_allowance, fs_seed, fs_alpha_computations],
                outputs=[fs_stats_output, fs_csv_train, fs_csv_test, fs_csv_optim, fs_json_output, fs_button],
            )

    with gr.Tab("Train"):
        train_csv = gr.File(label="Training CSV")
        train_target = gr.Textbox(label="Target column", value="target")
        train_classifier = gr.Dropdown(
            choices=["Random Forest", "SVM", "Logistic Regression"],
            value="Random Forest",
            label="Classifier",
        )
        train_model_name = gr.Textbox(label="Trained model filename", value="model.joblib")
        train_metrics_name = gr.Textbox(label="Training metrics filename", value="train_metrics.json")
        train_seed = gr.Number(label="Seed", value=42, precision=0)
        train_button = gr.Button("Train")
        gr.Markdown("## Output\nThe output will include the trained model and a JSON file containing statistics about the training process.")
        train_metrics_output_json = gr.JSON(label="Training metrics")
        train_model_output = gr.File(label="Trained model")
        train_metrics_output = gr.File(label="Training metrics JSON")

        train_button.click(
            fn=run_train,
            inputs=[train_csv, train_target, train_classifier, train_seed, train_model_name, train_metrics_name],
            outputs=[train_metrics_output_json, train_model_output, train_metrics_output, train_button],
        )

    with gr.Tab("Predict"):
        predict_csv = gr.File(label="Test CSV")
        predict_target = gr.Textbox(label="Target column", value="target")
        predict_model = gr.File(label="Trained model (.joblib)", file_types=[".joblib"])
        predict_button = gr.Button("Predict")
        gr.Markdown("## Output\nThe output will include a CSV file with the predictions and a JSON file containing statistics about the classification process.")
        predict_stats_output_json = gr.JSON(label="Classification statistics")
        predict_csv_output = gr.File(label="Predictions CSV")
        predict_stats_output = gr.File(label="Classification statistics JSON")

        predict_button.click(
            fn=run_predict,
            inputs=[predict_csv, predict_target, predict_model],
            outputs=[predict_stats_output_json, predict_csv_output, predict_stats_output, predict_button],
        )


if __name__ == "__main__":
    demo.launch()
