import os
import torch
import json
import PIL
from torch.utils.data import Dataset, DataLoader
from typing import *
from PIL import Image
from pathlib import Path
from torchvision import transforms


class ImagePreprocess:

    def __init__(self,
                 img_size: int = 224,
                 mean: list = None,
                 std: list = None):
        if mean is None:
            mean = [0.485, 0.456, 0.406]
        if std is None:
            std = [0.229, 0.224, 0.225]

        self.transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, image=None, *args, **kwargs):
        image = self.transform(img=image)
        return image


class StateImageDataset(Dataset):

    def __init__(self, data_path: Union[Path, str]):
        """
        Args:
            data_path (string): Directory with all the images and action.jsonl
        """
        self.data_path = Path(data_path)
        self.data_jsonl = self.data_path / "state.jsonl"
        with open(self.data_jsonl, mode="r", encoding="utf8") as jsonl_reader:
            self.dataset = jsonl_reader.readlines()

        self.images = []
        for image_info in self.dataset:
            read_json = json.loads(image_info.strip("\n"))
            label = read_json["state"]
            image_name = read_json["image_name"]
            self.images.append((label, image_name))

        self.transform = ImagePreprocess()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        label, image_name = self.images[idx]
        image = Image.open(self.data_path / image_name).convert("RGB")

        image = self.transform(image)

        return image, label


class AutoGameImageDataset(Dataset):

    def __init__(self, data_path: Union[Path, str]):
        """
        Args:
            data_path (string): Directory with all the images and action.jsonl
        """
        self.data_path = Path(data_path)
        self.action_data_jsonl = self.data_path / "action.jsonl"
        self.state_data_jsonl = self.data_path / "state.jsonl"
        with open(self.action_data_jsonl, mode="r", encoding="utf8") as jsonl_reader:
            self.action_dataset = jsonl_reader.readlines()

        with open(self.state_data_jsonl, mode="r", encoding="utf8") as jsonl_reader:
            self.state_dataset = jsonl_reader.readlines()

        self.action_images = []
        for index, image_info in enumerate(self.action_dataset):
            read_json = json.loads(image_info.strip("\n"))
            label = read_json["label"]
            image_name = str(read_json["image_name"])
            self.action_images.append((label, image_name))

        self.states_images = []
        for index, image_info in enumerate(self.state_dataset):
            read_json = json.loads(image_info.strip("\n"))
            label = read_json["state"]
            image_name = str(read_json["image_name"])
            self.states_images.append((label, image_name))

        # sorted by image index
        self.action_images = sorted(self.action_images, key=lambda x: int(x[1].split(".")[0]))
        self.states_images = sorted(self.states_images, key=lambda x: int(x[1].split(".")[0]))

        # Resize, To Tensor, Normalize
        self.transform = ImagePreprocess()

    def __len__(self):
        return len(self.action_images)

    def __getitem__(self, idx):
        action_label, image_name = self.action_images[idx]
        state_label = self.states_images[idx][0]

        image = Image.open(self.data_path / image_name).convert("RGB")

        image = self.transform(image)

        return image, action_label, state_label


def auto_game_collate_fn(batch):
    images, labels, state_labels = zip(*batch)
    images = torch.stack(images, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    state_labels = torch.tensor(state_labels, dtype=torch.long)
    return images, labels, state_labels


def state_collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return images, labels


# if __name__ == "__main__":
#     dataset = AutoGameImageDataset("../data/2024-08-26_19-02-41")
#     data_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4, collate_fn=auto_game_collate_fn)
#
#     for images, labels, state_labels in data_loader:
#         print(f'Batch of images shape: {images.shape}')
#         print(f'Batch of labels shape: {labels}, {labels.shape}')
#         print(f'Batch of labels shape: {state_labels}, {state_labels.shape}')
#         break
