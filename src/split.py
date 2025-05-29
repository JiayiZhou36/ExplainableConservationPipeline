from sklearn.model_selection import train_test_split


def split_dataset(images, test_size=0.2, random_seed=42):
    return train_test_split(images, test_size=test_size, random_state=random_seed)
