from PIL import Image
from typing import Optional, Callable, Tuple, Any
import torch
import pandas as pd
import os


def pil_loader(path: str) -> Image.Image:
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class KFood27(torch.utils.data.Dataset):
    meta_file = 'meta.csv'
    mean = (0.19671048, 0.16851181, 0.13641296)
    std = (0.10164582, 0.08350691, 0.0691753)

    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader = pil_loader,
    ) -> None:
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        dataframe = pd.read_csv(os.path.join(self.root, self.meta_file))
        self.images = dataframe.file_path.tolist()
        self.targets = dataframe.coarse_label.tolist()

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img = self.loader(os.path.join(self.root, self.images[index]))
        target = self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.targets)


class KFood150(torch.utils.data.Dataset):
    meta_file = 'meta.csv'
    mean = (0.19671048, 0.16851181, 0.13641296)
    std = (0.10164582, 0.08350691, 0.0691753)

    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader = pil_loader,
    ) -> None:
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        dataframe = pd.read_csv(os.path.join(self.root, self.meta_file))
        self.images = dataframe.file_path.tolist()
        self.targets = dataframe.fine_label.tolist()

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img = self.loader(os.path.join(self.root, self.images[index]))
        target = self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.targets)