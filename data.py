import os
import torch
import torchvision
import numpy as np
from PIL import Image
from config import DATA_PATH


def loader(batch_size, img_size=(224, 224), shuffle=True):
    dataset = ImgDataset(img_size=img_size)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)


class ImgDataset(torch.utils.data.Dataset):
    def __init__(self, img_size=(224, 224), path=DATA_PATH):
        self.img_size = img_size
        with open(os.path.join(path, "sets/train.txt")) as f:
            lines = f.readlines()
            self.img_files = map(
                    lambda l: os.path.join(path, "images", l.replace('\n', '.jpg')),
                    lines)
            self.seg_files = map(
                    lambda l: os.path.join(path, "masks", l.replace('\n', '.png')),
                    lines)

            def size_checker(f):
                img = Image.open(f)
                w, h = img.size
                return w >= img_size[0] & h >= img_size[1]

            self.img_files = filter(size_checker, self.img_files)
            self.seg_files = filter(size_checker, self.seg_files)
            self.img_files = list(self.img_files)
            self.seg_files = list(self.seg_files)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img = Image.open(self.img_files[idx])
        seg = Image.open(self.seg_files[idx])

        t, l, h, w = torchvision.transforms.RandomCrop.get_params(img, output_size=self.img_size)
        img = torchvision.transforms.functional.crop(img, t, l, h, w)
        seg = torchvision.transforms.functional.crop(seg, t, l, h, w)

        toTensor = torchvision.transforms.ToTensor()
        img = toTensor(img)

        seg = torch.tensor(np.array(seg), dtype=torch.long)
        seg[seg==255] = 0  

        return img, seg
