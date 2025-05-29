from torchvision import transforms


def get_transforms(config):
    transform_list = []
    if config["resize"]:
        transform_list.append(transforms.Resize(config["resize"]))
    if config["normalize"]:
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize((0.5,), (0.5,)))
    return transforms.Compose(transform_list)


def preprocess_images(images, transform):
    return [transform(img) for img in images]
