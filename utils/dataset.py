from datasets import load_dataset
import torchvision.transforms as transforms

dataset = load_dataset("cats_vs_dogs", split="train")
dataset.set_format(type="torch")


def resize(dataset):
    dataset["image"] = [
        transforms.functional.resize(image, size=(224, 224))
        for image in dataset["image"]
    ]
    return dataset


dataset.set_transform(resize)
