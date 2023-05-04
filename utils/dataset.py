from datasets import load_dataset
import torchvision.transforms as transforms


def resize(dataset):
    convert_tensor = transforms.ToTensor()
    image_resized = [convert_tensor(transforms.functional.resize(image, size=(224, 224))) for image in dataset['image']]
    new_dataset = [(image, labels) for (image, labels) in zip(image_resized, dataset["labels"]) if image.size()[0] == 3]
    image, labels = list(map(list, zip(*new_dataset)))
    dataset["image"] = image
    dataset["labels"] = labels
    return dataset

