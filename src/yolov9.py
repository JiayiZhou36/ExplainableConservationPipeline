# Use ultralytics library
from ultralytics import YOLO


def train_yolo(config, train_data):
    model = YOLO(config["weights"])
    model.train(
        data=train_data,
        epochs=config["epochs"],
        batch=config["batch_size"],
        lr0=config["learning_rate"],
    )
    model.save(config["model_dir"])
    return model
