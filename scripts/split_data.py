# scripts/split_data.py
import os
import argparse
import glob
import random
import shutil
from sklearn.model_selection import train_test_split
from .utils import logger  # Use the configured logger from utils


def split_data(
    source_image_dir,
    source_label_dir,
    output_base_dir,
    train_ratio,
    val_ratio,
    random_seed=42,
):
    """
    Splits data into train, validation, and test sets.
    Assumes images and corresponding YOLO labels (.txt) share the same base filename.
    """
    logger.info("Starting data splitting...")
    random.seed(random_seed)

    # Get all image files (jpg, png, jpeg)
    image_extensions = ["*.[jJ][pP][gG]", "*.[jJ][pP][eE][gG]", "*.[pP][nN][gG]"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(source_image_dir, ext)))

    if not image_files:
        logger.error(f"No image files found in {source_image_dir}. Cannot split data.")
        # Create dummy files to allow pipeline to proceed for demonstration if source_image_dir is empty
        # This is mainly for the GitHub Action initial run where data/raw/images might be empty
        # before data_ingestion creates dummy files.
        # However, preprocess.py should have created processed files. If that output is empty, this will fail.
        # Let's assume preprocess.py created at least one dummy processed file.
        # If `source_image_dir` (e.g. `data/processed/images`) is empty, it's an issue.
        # The dummy file creation in data_ingestion.py and subsequent processing in preprocess.py
        # should ensure at least one file exists here if the pipeline runs from scratch.
        # If `data/raw/images/drone_image_001.jpg` was created by GH Actions,
        # then `data/processed/images/drone_image_001.jpg` should exist.

        # Check if the single dummy file from GH actions exists (after preprocessing)
        dummy_processed_img_path = os.path.join(source_image_dir, "drone_image_001.jpg")
        dummy_processed_label_path = os.path.join(
            source_label_dir, "drone_image_001.txt"
        )

        if os.path.exists(dummy_processed_img_path):
            logger.info(
                f"Found dummy processed image: {dummy_processed_img_path}. Using it for split."
            )
            image_files.append(dummy_processed_img_path)
            # Ensure dummy label also exists for it
            if not os.path.exists(dummy_processed_label_path):
                logger.warning(
                    f"Dummy label {dummy_processed_label_path} not found. Creating empty one."
                )
                os.makedirs(source_label_dir, exist_ok=True)
                with open(dummy_processed_label_path, "w") as f:
                    f.write("")  # Empty label file
        else:
            logger.error(
                f"No images found in {source_image_dir}, including the expected dummy file. Aborting split."
            )
            return

    logger.info(
        f"Found {len(image_files)} total images in {source_image_dir} for splitting."
    )

    # Create unique list of base filenames (without extension)
    basenames = sorted(
        list(set([os.path.splitext(os.path.basename(f))[0] for f in image_files]))
    )

    if not basenames:
        logger.error("No basenames extracted from image files. Cannot proceed.")
        return

    logger.info(f"Total unique items (basenames) to split: {len(basenames)}")

    # Split basenames
    train_basenames, temp_basenames = train_test_split(
        basenames, test_size=(1.0 - train_ratio), random_state=random_seed
    )

    if (
        not temp_basenames
    ):  # Handles cases with very few files where temp_basenames might be empty
        if (
            len(basenames) > 0 and len(basenames) <= 2 and train_ratio >= 0.5
        ):  # if 1 or 2 files, put all in train, or one in train one in val
            logger.warning("Very few files. Adjusting split logic.")
            train_basenames = basenames[:1]
            val_basenames = basenames[1:] if len(basenames) > 1 else []
            test_basenames = []
        else:
            logger.error(
                "Not enough data for validation/test split after initial train split."
            )
            val_basenames, test_basenames = [], []
    else:
        # Calculate new val_ratio relative to the remaining temp_basenames
        # E.g. if total 100, train_ratio 0.7 (70 files), temp is 30 files.
        # If val_ratio (original) is 0.15 (15 files), then new_val_ratio for temp is 15/30 = 0.5
        if (1.0 - train_ratio) == 0:  # if train_ratio is 1.0
            relative_val_ratio = 0
        else:
            relative_val_ratio = val_ratio / (1.0 - train_ratio)

        if (
            relative_val_ratio >= 1.0 and len(temp_basenames) > 0
        ):  # if val takes all remaining or more
            val_basenames = temp_basenames
            test_basenames = []
        elif relative_val_ratio <= 0 and len(temp_basenames) > 0:  # if val takes none
            val_basenames = []
            test_basenames = temp_basenames
        elif not temp_basenames:
            val_basenames = []
            test_basenames = []
        else:  # Standard case
            val_basenames, test_basenames = train_test_split(
                temp_basenames,
                test_size=(1.0 - relative_val_ratio),
                random_state=random_seed,
            )

    logger.info(f"Train set size: {len(train_basenames)}")
    logger.info(f"Validation set size: {len(val_basenames)}")
    logger.info(f"Test set size: {len(test_basenames)}")

    # Define output directories
    sets = {"train": train_basenames, "val": val_basenames, "test": test_basenames}

    for set_name, b_names in sets.items():
        if not b_names and set_name in [
            "train",
            "val",
        ]:  # Allow test set to be empty if val takes all
            logger.warning(
                f"{set_name} set is empty. This might be an issue for training/evaluation."
            )
            # Create directories anyway for consistency
            os.makedirs(
                os.path.join(output_base_dir, set_name, "images"), exist_ok=True
            )
            os.makedirs(
                os.path.join(output_base_dir, set_name, "labels"), exist_ok=True
            )
            continue
        elif not b_names and set_name == "test":
            logger.info(
                "Test set is empty. This might be intended if all data used for train/val."
            )
            os.makedirs(
                os.path.join(output_base_dir, set_name, "images"), exist_ok=True
            )
            os.makedirs(
                os.path.join(output_base_dir, set_name, "labels"), exist_ok=True
            )
            continue

        output_img_subdir = os.path.join(output_base_dir, set_name, "images")
        output_lbl_subdir = os.path.join(output_base_dir, set_name, "labels")
        os.makedirs(output_img_subdir, exist_ok=True)
        os.makedirs(output_lbl_subdir, exist_ok=True)

        logger.info(f"Copying files for {set_name} set...")
        for bn in tqdm(b_names, desc=f"Copying {set_name} files"):
            # Find the first matching image file for the basename (handles multiple extensions)
            img_src_path = None
            for ext_pattern in ["*.jpg", "*.jpeg", "*.png"]:  # Common extensions
                potential_paths = glob.glob(
                    os.path.join(
                        source_image_dir, bn + os.path.splitext(ext_pattern)[1]
                    )
                )
                if potential_paths:
                    img_src_path = potential_paths[0]
                    break

            if not img_src_path or not os.path.exists(img_src_path):
                logger.warning(
                    f"Source image for basename {bn} not found in {source_image_dir}. Skipping."
                )
                continue

            lbl_src_path = os.path.join(source_label_dir, bn + ".txt")

            # Copy image
            img_dest_path = os.path.join(
                output_img_subdir, os.path.basename(img_src_path)
            )
            shutil.copy2(img_src_path, img_dest_path)

            # Copy label if it exists
            if os.path.exists(lbl_src_path):
                lbl_dest_path = os.path.join(output_lbl_subdir, bn + ".txt")
                shutil.copy2(lbl_src_path, lbl_dest_path)
            else:
                logger.warning(
                    f"Label file for {bn} not found at {lbl_src_path}. Image {os.path.basename(img_src_path)} will not have a corresponding label in {set_name} set."
                )
                # Create an empty label file for consistency if model expects it,
                # but this usually indicates a problem with the raw data or preprocessing.
                # For YOLO, an image without a label file means no objects in that image.
                # So, creating an empty label file is the correct behavior.
                lbl_dest_path = os.path.join(output_lbl_subdir, bn + ".txt")
                with open(lbl_dest_path, "w") as f:
                    pass  # Create empty file
                logger.info(
                    f"Created empty label file for {bn} in {set_name} set as source was missing."
                )

    logger.info("Data splitting completed.")
    logger.info(f"Train data: {output_base_dir}/train")
    logger.info(f"Validation data: {output_base_dir}/val")
    logger.info(f"Test data: {output_base_dir}/test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Data Splitting Script for MLOps Pipeline"
    )
    # These source directories should point to where preprocess.py saved its output.
    # In the YAML, this would be data/processed/images and data/processed/labels.
    parser.add_argument(
        "--source_image_dir",
        type=str,
        required=True,
        help="Directory of preprocessed images.",
    )
    parser.add_argument(
        "--source_label_dir",
        type=str,
        required=True,
        help="Directory of preprocessed labels.",
    )
    # This output_base_dir is data/processed (which contains train/val/test subdirs)
    parser.add_argument(
        "--output_base_dir",
        type=str,
        required=True,
        help="Base directory to save train/val/test splits.",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="Proportion of data for training.",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="Proportion of data for validation.",
    )
    # Test ratio is implicitly 1.0 - train_ratio - val_ratio

    args = parser.parse_args()

    if args.train_ratio + args.val_ratio > 1.0:
        logger.error("Sum of train_ratio and val_ratio cannot exceed 1.0.")
    else:
        split_data(
            args.source_image_dir,
            args.source_label_dir,
            args.output_base_dir,
            args.train_ratio,
            args.val_ratio,
        )
