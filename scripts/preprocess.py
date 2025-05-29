# scripts/preprocess.py
import os
import argparse
import glob
import cv2  # OpenCV for image processing
from tqdm import tqdm  # For progress bars
from .utils import logger  # Use the configured logger from utils


def preprocess_image(image_path, output_image_path, img_size):
    """
    Preprocesses a single image: reads, resizes, and saves it.
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            logger.warning(f"Could not read image: {image_path}. Skipping.")
            return False

        # Resize image (maintaining aspect ratio by padding if necessary, or simple resize)
        # For simplicity, we'll do a direct resize. YOLO often handles padding internally or expects square images.
        resized_image = cv2.resize(image, (img_size, img_size))

        # Normalization is typically handled by the model/training framework (e.g., dividing by 255.0)
        # So we just save the resized image.

        cv2.imwrite(output_image_path, resized_image)
        return True
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {e}")
        return False


def preprocess_annotations(
    annotation_path, output_label_path, img_size, original_img_shape=None
):
    """
    Preprocesses a single annotation file.
    For YOLO format, if images are resized, annotations might not need changes if they are relative (0-1).
    However, if absolute pixel values were used and converted to relative, resizing matters.
    Assuming YOLO format (class_id, cx, cy, w, h) is already relative to image dimensions.
    If augmentations like cropping or padding that change aspect ratio significantly are applied,
    bounding boxes need adjustment. For simple resize to square, relative coordinates are fine.
    This function will mostly copy the annotation if it's already in YOLO format.
    """
    try:
        # For this example, we assume annotations are already in the correct relative YOLO format.
        # If complex augmentations were applied, this step would be more involved.
        # For now, just copy the annotation file.
        if os.path.exists(annotation_path):
            with open(annotation_path, "r") as f_in, open(
                output_label_path, "w"
            ) as f_out:
                f_out.write(f_in.read())
            return True
        else:
            logger.warning(f"Annotation file not found: {annotation_path}. Skipping.")
            return False
    except Exception as e:
        logger.error(f"Error processing annotation {annotation_path}: {e}")
        return False


def run_preprocessing(
    image_dir, annotation_dir, output_image_dir, output_label_dir, img_size
):
    """
    Main preprocessing function.
    Iterates through images and annotations, preprocesses them, and saves to output directories.
    """
    logger.info("Starting preprocessing...")
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    image_files = (
        glob.glob(os.path.join(image_dir, "*.[jJ][pP][gG]"))
        + glob.glob(os.path.join(image_dir, "*.[jJ][pP][eE][gG]"))
        + glob.glob(os.path.join(image_dir, "*.[pP][nN][gG]"))
    )

    if not image_files:
        logger.warning(f"No images found in {image_dir}. Preprocessing cannot proceed.")
        return

    logger.info(f"Found {len(image_files)} images to preprocess.")

    for image_path in tqdm(image_files, desc="Preprocessing Images"):
        basename = os.path.basename(image_path)
        name, ext = os.path.splitext(basename)

        output_image_path = os.path.join(output_image_dir, basename)
        annotation_filename = name + ".txt"  # Assuming YOLO annotation format
        annotation_path = os.path.join(annotation_dir, annotation_filename)
        output_label_path = os.path.join(output_label_dir, annotation_filename)

        # Preprocess image
        if not preprocess_image(image_path, output_image_path, img_size):
            continue  # Skip to next image if preprocessing failed

        # Preprocess annotation (copy for now)
        if os.path.exists(annotation_path):
            preprocess_annotations(annotation_path, output_label_path, img_size)
        else:
            logger.warning(
                f"Annotation file {annotation_filename} not found for image {basename}. An empty label file might be created by data splitting if this image is used."
            )
            # Create an empty label file if annotation is missing, so that split_data doesn't break
            # if it expects a label file for every image.
            # However, models typically require annotations for training.
            # For now, we'll just log. The split script should handle this.

    logger.info("Preprocessing completed.")
    logger.info(f"Processed images saved to: {output_image_dir}")
    logger.info(f"Processed labels saved to: {output_label_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocessing Script for MLOps Pipeline"
    )
    parser.add_argument(
        "--image_dir", type=str, required=True, help="Directory of raw images."
    )
    parser.add_argument(
        "--annotation_dir",
        type=str,
        required=True,
        help="Directory of raw annotations.",
    )
    parser.add_argument(
        "--output_image_dir",
        type=str,
        required=True,
        help="Directory to save processed images.",
    )
    parser.add_argument(
        "--output_label_dir",
        type=str,
        required=True,
        help="Directory to save processed labels.",
    )
    parser.add_argument(
        "--img_size", type=int, default=640, help="Target image size (square)."
    )

    args = parser.parse_args()

    # In the GitHub Actions YAML, preprocess.py is called before split_data.py.
    # preprocess.py will put all processed images/labels into a general output_image_dir/output_label_dir.
    # Then, split_data.py will take from these general processed directories and create train/val/test splits.
    # This means the YAML should be:
    # 1. Ingestion (raw_data_dir)
    # 2. Preprocessing (raw_data_dir -> temp_processed_dir)
    # 3. Splitting (temp_processed_dir -> final data/processed/train, data/processed/val, data/processed/test)
    # Let's adjust the YAML's call to preprocess.py and split_data.py accordingly.
    # For now, this script assumes it's outputting to a general processed folder.
    # The YAML currently has `output_image_dir data/processed/images` which is fine.
    # `split_data.py` will then take `data/processed/images` and `data/processed/labels` as source.

    run_preprocessing(
        args.image_dir,
        args.annotation_dir,
        args.output_image_dir,
        args.output_label_dir,
        args.img_size,
    )
