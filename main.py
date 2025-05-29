import yaml
from src import data_loader, preprocess, split, yolov9, evaluate, explain, logger

with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

log = logger.setup_logger(
    config["logging"]["log_dir"], config["logging"]["experiment_name"]
)

images = data_loader.load_images(config["paths"]["raw_data"])
transform = preprocess.get_transforms(config["preprocessing"])
processed = preprocess.preprocess_images(images, transform)
train_data, test_data = split.split_dataset(
    processed, test_size=config["split"]["test_size"]
)

log.info("Training model...")
model = yolov9.train_yolo(config["model"], train_data)

log.info("Evaluating model...")
results = evaluate.evaluate_model(model, test_data)

log.info("Applying explainability...")
if "gradcam" in config["explainability"]["techniques"]:
    explain.apply_gradcam(model, test_data, config["paths"]["output_dir"])
if "shap" in config["explainability"]["techniques"]:
    explain.apply_shap(model, test_data, config["paths"]["output_dir"])
