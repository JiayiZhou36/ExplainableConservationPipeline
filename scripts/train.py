# scripts/train.py
import os
import argparse
import subprocess  # To call YOLOv9 training script
import mlflow
import time
import glob
import shutil  # For creating dummy model
from .utils import logger, get_mlflow_experiment_id, create_yolo_data_yaml

# Define dummy class names and number of classes for the data YAML
# In a real scenario, this would come from your dataset or a config file.
DUMMY_CLASS_NAMES = ["object"]  # Replace with your actual class names
DUMMY_NUM_CLASSES = len(DUMMY_CLASS_NAMES)


def train_model(
    data_config_path,
    img_size,
    batch_size,
    epochs,
    weights,
    output_model_path,
    project_name,
    run_name,
):
    """
    Simulates YOLOv9 model training.
    In a real scenario, this would involve calling the actual YOLOv9 training script
    (e.g., from a cloned YOLOv9 repository).
    """
    logger.info("Starting model training (simulation)...")
    experiment_id = get_mlflow_experiment_id(project_name)

    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
        run_id = run.info.run_id
        logger.info(f"MLflow Run ID: {run_id}")
        logger.info(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
        logger.info(f"MLflow Artifact URI: {mlflow.get_artifact_uri()}")

        # Log parameters
        mlflow.log_param("data_config", data_config_path)
        mlflow.log_param("img_size", img_size)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("initial_weights", weights)
        mlflow.log_param("yolo_version", "yolov9_simulated")

        # --- Actual YOLOv9 training command would go here ---
        # This is a placeholder. You would need to have the YOLOv9 repository
        # cloned and set up, then call its train.py script.
        # Example (conceptual, actual command depends on YOLOv9 repo structure):
        # yolo_train_cmd = [
        #     'python', 'path/to/yolov9/train.py',
        #     '--weights', weights,
        #     '--data', data_config_path,
        #     '--epochs', str(epochs),
        #     '--batch-size', str(batch_size),
        #     '--img-size', str(img_size),
        #     '--project', os.path.join('runs/train', project_name), # YOLOv9 saves outputs here
        #     '--name', run_name,
        #     '--device', '0' # or 'cpu'
        # ]
        # logger.info(f"Executing YOLOv9 training command: {' '.join(yolo_train_cmd)}")
        # try:
        #     process = subprocess.Popen(yolo_train_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        #     for line in iter(process.stdout.readline, ''):
        #         logger.info(line.strip()) # Log YOLOv9 output
        #         # Parse metrics from output if possible and log to MLflow
        #         # e.g., if line contains "mAP@.5": mlflow.log_metric("mAP_0.5", value, step=epoch)
        #     process.stdout.close()
        #     return_code = process.wait()
        #     if return_code != 0:
        #         logger.error(f"YOLOv9 training failed with return code {return_code}")
        #         raise Exception(f"YOLOv9 training failed. Check logs.")
        # except FileNotFoundError:
        #     logger.error("YOLOv9 training script not found. Ensure YOLOv9 is correctly set up.")
        #     raise
        # logger.info("YOLOv9 training process completed.")

        # --- Simulation of training ---
        logger.info("Simulating training process...")
        for epoch in range(1, epochs + 1):
            time.sleep(1)  # Simulate time taken per epoch
            # Simulate metrics
            train_loss = 1.0 / epoch + 0.1 * random.random()
            val_loss = 1.2 / epoch + 0.1 * random.random()
            mAP_0_5 = 0.1 * epoch * random.uniform(0.8, 1.0)  # Simulated mAP
            if mAP_0_5 > 1.0:
                mAP_0_5 = 1.0

            logger.info(
                f"Epoch {epoch}/{epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, mAP@0.5={mAP_0_5:.4f}"
            )
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("mAP_0.5", mAP_0_5, step=epoch)

        # Simulate saving the best model
        # In a real YOLOv9 run, the best model is saved in runs/train/project_name/run_name/weights/best.pt
        # We will create a dummy model file.
        os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
        with open(output_model_path, "w") as f:
            f.write(
                f"Simulated YOLOv9 model for project {project_name}, run {run_name}\n"
            )
            f.write(f"Trained for {epochs} epochs.\n")
        logger.info(f"Simulated model saved to {output_model_path}")

        # Log the model artifact to MLflow
        # If using actual YOLOv9, you'd log the 'best.pt' from its output directory.
        # Example: yolo_output_dir = os.path.join('runs/train', project_name, run_name)
        # yolo_best_model_path = os.path.join(yolo_output_dir, 'weights/best.pt')
        # if os.path.exists(yolo_best_model_path):
        #    mlflow.log_artifact(yolo_best_model_path, artifact_path="model")
        # else:
        #    logger.warning(f"YOLOv9 best model not found at {yolo_best_model_path}")

        # Log our dummy model
        mlflow.log_artifact(output_model_path, artifact_path="model")

        # Log data.yaml as an artifact
        if os.path.exists(data_config_path):
            mlflow.log_artifact(data_config_path, artifact_path="config")
        else:
            logger.warning(f"Data config {data_config_path} not found for logging.")

        # (Optional) Log other artifacts like plots if YOLOv9 generates them (e.g., confusion matrix, P-R curve)
        # yolo_plots = glob.glob(os.path.join(yolo_output_dir, "*.png"))
        # for plot_file in yolo_plots:
        #    mlflow.log_artifact(plot_file, artifact_path="training_plots")

        logger.info(
            "Model training simulation completed and artifacts logged to MLflow."
        )
        mlflow.set_tag("training_status", "completed_simulation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="YOLOv9 Model Training Script (Simulated)"
    )
    parser.add_argument(
        "--data_config_path",
        type=str,
        required=True,
        help="Path to the YOLO data YAML file (e.g., drone_data.yaml).",
    )
    parser.add_argument(
        "--img_size", type=int, default=640, help="Image size for training."
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for training."
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs."
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="yolov9-c.pt",
        help="Initial weights path or official YOLOv9 model name (e.g., yolov9-c.pt, yolov9-e.pt). For simulation, this is just a parameter.",
    )
    parser.add_argument(
        "--output_model_path",
        type=str,
        required=True,
        help="Path to save the trained model artifact.",
    )
    parser.add_argument(
        "--project_name",
        type=str,
        default="drone_detection",
        help="MLflow project/experiment name.",
    )
    parser.add_argument(
        "--run_name", type=str, default="yolov9_training_run", help="MLflow run name."
    )

    # Arguments for creating drone_data.yaml
    # These paths should come from the output of split_data.py
    parser.add_argument(
        "--train_images_dir",
        type=str,
        default="data/processed/train/images",
        help="Path to training images directory.",
    )
    parser.add_argument(
        "--val_images_dir",
        type=str,
        default="data/processed/val/images",
        help="Path to validation images directory.",
    )
    parser.add_argument(
        "--test_images_dir",
        type=str,
        default="data/processed/test/images",
        help="Path to test images directory (for data.yaml).",
    )

    args = parser.parse_args()

    # Create the data.yaml file needed by YOLO
    # Ensure the paths in data.yaml are absolute or relative to where YOLO's train.py is executed
    # For GitHub Actions, absolute paths are safer.
    # The YAML file itself will be placed at args.data_config_path

    # Check if directories exist, create dummy ones if not (for standalone script run)
    os.makedirs(args.train_images_dir, exist_ok=True)
    os.makedirs(args.val_images_dir, exist_ok=True)
    os.makedirs(args.test_images_dir, exist_ok=True)

    create_yolo_data_yaml(
        output_path=args.data_config_path,
        train_img_dir=os.path.abspath(args.train_images_dir),
        val_img_dir=os.path.abspath(args.val_images_dir),
        test_img_dir=os.path.abspath(
            args.test_images_dir
        ),  # test_img_dir is for val.py/detect.py in YOLO
        class_names=DUMMY_CLASS_NAMES,
        num_classes=DUMMY_NUM_CLASSES,
    )

    # Quick check if dummy files exist for training, if not, create one in train set
    # This ensures that even with minimal data, the paths in drone_data.yaml are valid.
    if not glob.glob(os.path.join(args.train_images_dir, "*")):
        logger.warning(
            f"Train images directory {args.train_images_dir} is empty. Creating a dummy image for YAML validation."
        )
        dummy_train_img = os.path.join(args.train_images_dir, "dummy_train_img.jpg")
        with open(dummy_train_img, "w") as f:
            f.write("")
    if not glob.glob(os.path.join(args.val_images_dir, "*")):
        logger.warning(
            f"Validation images directory {args.val_images_dir} is empty. Creating a dummy image for YAML validation."
        )
        dummy_val_img = os.path.join(args.val_images_dir, "dummy_val_img.jpg")
        with open(dummy_val_img, "w") as f:
            f.write("")

    train_model(
        data_config_path=args.data_config_path,
        img_size=args.img_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        weights=args.weights,
        output_model_path=args.output_model_path,
        project_name=args.project_name,
        run_name=args.run_name,
    )
