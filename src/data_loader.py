import os
from PIL import Image


def load_images(data_dir):
    images = []
    for file in os.listdir(data_dir):
        if file.endswith(".jpg") or file.endswith(".png"):
            images.append(Image.open(os.path.join(data_dir, file)))
    return images
