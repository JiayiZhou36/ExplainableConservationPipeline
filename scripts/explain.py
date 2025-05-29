# scripts/explain.py
import os
import argparse
import mlflow
import cv2  # For image handling
import numpy as np  # For creating dummy heatmaps
import random
from .utils import logger, get_mlflow_experiment_id


def generate_simulated_heatmap(image):
    """
    Simulates generating a heatmap (e.g., Grad-CAM) for an image.
    Returns a dummy heatmap.
    """
    if image is None:
        return None
    heatmap = np.random.rand(image.shape[0], image.shape[1]) * 255
    heatmap = heatmap.astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Blend heatmap with original image
    overlayed_image = cv2.addWeighted(image, 0.6, heatmap_color, 0.4, 0)
    return overlayed_image


def run_explainability(model_path, image_path, output_dir, project_name, run_name):
    """
    Simulates applying explainability techniques.
    For object detection, this could be Grad-CAM on detected objects, etc.
    """
    logger.info("Starting explainability analysis (simulation)...")

    experiment_id = get_mlflow_experiment_id(project_name)

    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
        explain_run_id = run.info.run_id
        logger.info(f"MLflow Explainability Run ID: {explain_run_id}")

        mlflow.log_param("model_path_explain", model_path)
        mlflow.log_param("source_image_path", image_path)
        mlflow.log_param("explain_method", "simulated_grad_cam")

        os.makedirs(output_dir, exist_ok=True)

        # Load the image
        if not os.path.exists(image_path):
            logger.error(f"Image for explanation not found: {image_path}")
            # Create a dummy image if not found, for pipeline to proceed
            logger.info(f"Creating a dummy image at {image_path} for explanation.")
            dummy_image_data = np.zeros(
                (100, 100, 3), dtype=np.uint8
            )  # Black 100x100 image
            cv2.imwrite(image_path, dummy_image_data)
            # return # Or raise error

        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Could not read image: {image_path}")
            mlflow.set_tag("explainability_status", "failed_image_load")
            return

        # --- Actual Explainability Logic would go here ---
        # This would involve loading the model, performing inference, then applying
        # a technique like Grad-CAM, LIME, SHAP, or visualizing attention.
        # For example, using a library like `captum` for PyTorch models.
        # This is highly model and task-specific.

        # Simulate generating a heatmap
        logger.info("Simulating Grad-CAM heatmap generation...")
        heatmap_overlay_image = generate_simulated_heatmap(image.copy())

        if heatmap_overlay_image is not None:
            explanation_output_path = os.path.join(
                output_dir, f"explanation_{os.path.basename(image_path)}"
            )
            cv2.imwrite(explanation_output_path, heatmap_overlay_image)
            logger.info(
                f"Simulated explanation (heatmap overlay) saved to: {explanation_output_path}"
            )
            mlflow.log_artifact(explanation_output_path, artifact_path="explanations")
        else:
            logger.warning("Failed to generate simulated heatmap.")

        # Simulate some textual explanation or feature importance
        simulated_explanation_text = {
            "image_analyzed": os.path.basename(image_path),
            "important_regions_summary": "Simulated: Model focused on central and upper-left regions.",
            "confidence_for_top_detection": f"{random.uniform(0.7, 0.95):.2f} (simulated)",
        }
        explanation_text_path = os.path.join(
            output_dir,
            f"explanation_summary_{os.path.splitext(os.path.basename(image_path))[0]}.txt",
        )
        with open(explanation_text_path, "w") as f:
            for key, value in simulated_explanation_text.items():
                f.write(f"{key}: {value}\n")
        logger.info(f"Simulated explanation summary saved to: {explanation_text_path}")
        mlflow.log_artifact(explanation_text_path, artifact_path="explanations")

        logger.info(
            "Explainability analysis simulation completed and artifacts logged to MLflow."
        )
        mlflow.set_tag("explainability_status", "completed_simulation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Explainability Script for MLOps Pipeline (Simulated)"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the trained model."
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to an image for explanation (e.g., from test set).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save explanation outputs.",
    )
    parser.add_argument(
        "--project_name",
        type=str,
        default="drone_detection",
        help="MLflow project/experiment name.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="yolov9_explainability_run",
        help="MLflow run name for this explainability task.",
    )

    args = parser.parse_args()

    # Ensure the model path exists (even if it's the dummy one from train.py)
    if not os.path.exists(args.model_path):
        logger.warning(
            f"Model path {args.model_path} does not exist. Creating a dummy one for explanation to proceed."
        )
        os.makedirs(
            os.path.dirname(args.model_path), exist_ok=True
        )  # Ensure directory exists
        with open(args.model_path, "w") as f:
            f.write("Dummy model for explanation.")

    run_explainability(
        model_path=args.model_path,
        image_path=args.image_path,
        output_dir=args.output_dir,
        project_name=args.project_name,
        run_name=args.run_name,
    )
