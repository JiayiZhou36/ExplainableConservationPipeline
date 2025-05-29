# scripts/utils.py
import mlflow
import os
import logging
import yaml

# Configure basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_mlflow_experiment_id(experiment_name):
    """
    Get the experiment ID for a given experiment name.
    Creates the experiment if it doesn't exist.
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        logger.info(f"Creating new MLflow experiment: {experiment_name}")
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
    logger.info(
        f"Using MLflow experiment ID: {experiment_id} for experiment: {experiment_name}"
    )
    return experiment_id


def load_config(config_path="config.yaml"):
    """Loads a YAML configuration file."""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found at {config_path}")
        return None
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {e}")
        return None


def create_yolo_data_yaml(
    output_path, train_img_dir, val_img_dir, test_img_dir, class_names, num_classes
):
    """
    Creates a YAML file in the format required by YOLO models for dataset configuration.
    """
    data_yaml_content = {
        "train": os.path.abspath(train_img_dir),
        "val": os.path.abspath(val_img_dir),
        "test": os.path.abspath(test_img_dir),
        "nc": num_classes,
        "names": class_names,
    }
    try:
        with open(output_path, "w") as f:
            yaml.dump(data_yaml_content, f, sort_keys=False)
        logger.info(f"YOLO data YAML created successfully at {output_path}")
        logger.info(f"Content: {data_yaml_content}")
    except Exception as e:
        logger.error(f"Failed to create YOLO data YAML at {output_path}: {e}")


if __name__ == "__main__":
    # Example usage (optional, for testing utils)
    # Ensure MLFLOW_TRACKING_URI is set, e.g., export MLFLOW_TRACKING_URI=file:./mlruns
    if os.getenv("MLFLOW_TRACKING_URI") is None:
        os.environ["MLFLOW_TRACKING_URI"] = "file:./mlruns_utils_test"
        logger.warning(
            "MLFLOW_TRACKING_URI not set, using default file:./mlruns_utils_test for utils test"
        )

    experiment_id = get_mlflow_experiment_id("UtilsTestExperiment")

    with mlflow.start_run(experiment_id=experiment_id, run_name="UtilTestRun") as run:
        logger.info(f"MLflow run started: {run.info.run_id}")
        mlflow.log_param("test_param", "test_value")
        mlflow.log_metric("test_metric", 0.99)
        # Create a dummy artifact
        with open("dummy_artifact.txt", "w") as f:
            f.write("This is a test artifact from utils.py.")
        mlflow.log_artifact("dummy_artifact.txt", artifact_path="utils_artifacts")
        os.remove("dummy_artifact.txt")
        logger.info("Test artifact logged.")
        logger.info("MLflow run completed.")

    # Example for creating data.yaml
    # These paths should be relative to where the training script will be run from, or absolute.
    # For GitHub Actions, os.path.abspath is good.
    # In the main pipeline, these paths would come from the split_data.py output.

    # Create dummy directories for the data.yaml example
    os.makedirs("temp_data/train/images", exist_ok=True)
    os.makedirs("temp_data/val/images", exist_ok=True)
    os.makedirs("temp_data/test/images", exist_ok=True)

    create_yolo_data_yaml(
        output_path="temp_drone_data.yaml",
        train_img_dir="temp_data/train/images",
        val_img_dir="temp_data/val/images",
        test_img_dir="temp_data/test/images",
        class_names=["drone", "person", "vehicle"],  # Example class names
        num_classes=3,
    )
    # Clean up dummy files and dirs
    if os.path.exists("temp_drone_data.yaml"):
        os.remove("temp_drone_data.yaml")
    if os.path.exists("temp_data"):
        import shutil

        shutil.rmtree("temp_data")
    if os.path.exists("mlruns_utils_test"):
        import shutil

        shutil.rmtree("mlruns_utils_test")

    logger.info("Utils script finished.")
