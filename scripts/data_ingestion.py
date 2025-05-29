# scripts/data_ingestion.py
import os
import argparse
import logging
import glob
import shutil  # For dummy file creation
from .utils import logger  # Use the configured logger from utils


def ingest_data(raw_image_dir, raw_annotation_dir, output_log):
    """
    Simulates data ingestion. In a real scenario, this could involve downloading
    from cloud storage, querying a database, or copying from a network location.
    For this example, it just checks if the directories exist and logs file counts.
    """
    logger.info("Starting data ingestion...")

    # Ensure directories exist
    os.makedirs(raw_image_dir, exist_ok=True)
    os.makedirs(raw_annotation_dir, exist_ok=True)

    # Create dummy files if they don't exist (for pipeline demonstration)
    if not glob.glob(os.path.join(raw_image_dir, "*")):
        logger.info(f"No images found in {raw_image_dir}. Creating dummy image.")
        dummy_img_path = os.path.join(raw_image_dir, "dummy_drone_image_001.jpg")
        with open(dummy_img_path, "w") as f:  # Create an empty file
            f.write("")
        logger.info(f"Created dummy image: {dummy_img_path}")

    if not glob.glob(os.path.join(raw_annotation_dir, "*")):
        logger.info(
            f"No annotations found in {raw_annotation_dir}. Creating dummy annotation."
        )
        dummy_ann_path = os.path.join(raw_annotation_dir, "dummy_drone_image_001.txt")
        with open(dummy_ann_path, "w") as f:
            f.write("0 0.5 0.5 0.1 0.1")  # class_id cx cy w h
        logger.info(f"Created dummy annotation: {dummy_ann_path}")

    image_files = glob.glob(os.path.join(raw_image_dir, "*"))
    annotation_files = glob.glob(os.path.join(raw_annotation_dir, "*"))

    num_images = len(image_files)
    num_annotations = len(annotation_files)

    logger.info(f"Found {num_images} images in {raw_image_dir}")
    logger.info(f"Found {num_annotations} annotations in {raw_annotation_dir}")

    if num_images == 0:
        logger.warning("No images found. Subsequent steps might fail.")
    if num_annotations == 0:
        logger.warning("No annotations found. Subsequent steps might fail.")

    # Basic validation: check if number of images and annotations match (optional, depends on setup)
    # For YOLO, typically one .txt file per image.
    image_basenames = {os.path.splitext(os.path.basename(f))[0] for f in image_files}
    annotation_basenames = {
        os.path.splitext(os.path.basename(f))[0] for f in annotation_files
    }

    if (
        image_basenames != annotation_basenames
        and num_images > 0
        and num_annotations > 0
    ):  # only warn if both are non-zero but mismatch
        logger.warning("Mismatch between image and annotation file basenames.")
        logger.warning(
            f"Images without annotations: {image_basenames - annotation_basenames}"
        )
        logger.warning(
            f"Annotations without images: {annotation_basenames - image_basenames}"
        )

    with open(output_log, "w") as f:
        f.write(f"Data Ingestion Summary:\n")
        f.write(f"Images found: {num_images}\n")
        f.write(f"Annotations found: {num_annotations}\n")
        if (
            image_basenames != annotation_basenames
            and num_images > 0
            and num_annotations > 0
        ):
            f.write(f"Warning: Mismatch detected between image and annotation files.\n")

    logger.info(f"Data ingestion summary logged to {output_log}")
    logger.info("Data ingestion completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Data Ingestion Script for MLOps Pipeline"
    )
    parser.add_argument(
        "--raw_image_dir",
        type=str,
        required=True,
        help="Directory containing raw images.",
    )
    parser.add_argument(
        "--raw_annotation_dir",
        type=str,
        required=True,
        help="Directory containing raw annotations.",
    )
    parser.add_argument(
        "--output_log", type=str, required=True, help="Path to save the ingestion log."
    )

    args = parser.parse_args()

    ingest_data(args.raw_image_dir, args.raw_annotation_dir, args.output_log)
