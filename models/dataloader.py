import os
import torch
import torchvision.transforms.functional as tf
import random
import numpy as np

from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset

from models import utils

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class MultiEpochsDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for _ in range(len(self)):
            yield next(self.iterator)

class _RepeatSampler(object):
    def __init__(self, sampler):
        self.sampler = sampler
    def __iter__(self):
        while True:
            yield from iter(self.sampler)

def is_image(src):
    return os.path.splitext(src)[1].lower() in ['.jpg', '.png', '.tif', '.ppm']

class Image2ImageLoader_resize(Dataset):
    def __init__(self, x_path, y_path, mode, **kwargs):
        self.mode = mode
        self.args = kwargs['args']
        self.image_mean = [0.485, 0.456, 0.406]
        self.image_std = [0.229, 0.224, 0.225]

        x_img_name = sorted(filter(is_image, os.listdir(x_path)))
        y_img_name = sorted(filter(is_image, os.listdir(y_path)))

        self.img_x_path = [os.path.join(x_path, f) for f in x_img_name]
        self.img_y_path = [os.path.join(y_path, f) for f in y_img_name]

        assert len(self.img_x_path) == len(self.img_y_path), 'Images in directory must have same file indices!!'

        print(f'{utils.Colors.LIGHT_RED}Mounting data on memory...{self.__class__.__name__}:{self.mode}{utils.Colors.END}')
        self.img_x = [Image.open(p).convert('RGB') for p in self.img_x_path]
        self.img_y = [Image.open(p).convert('L') for p in self.img_y_path]

    def transform(self, image, target):
        resize_h = int(self.args.input_size[0])
        resize_w = int(self.args.input_size[1])
        image = tf.resize(image, [resize_h, resize_w])
        target = tf.resize(target, [resize_h, resize_w], interpolation=InterpolationMode.NEAREST)

        if self.mode != 'validation':
            random_gen = random.Random()
            if getattr(self.args, 'transform_hflip', False) and (random_gen.random() < 0.5):
                image = tf.hflip(image)
                target = tf.hflip(target)
            if getattr(self.args, 'transform_jitter', False) and (random_gen.random() < 0.8):
                transform_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
                image = transform_jitter(image)
            if getattr(self.args, 'transform_blur', False) and (random_gen.random() < 0.5):
                kernel_size = int((random.random() * 10 + 2.5).__round__())
                if kernel_size % 2 == 0: kernel_size -= 1
                transform_blur = transforms.GaussianBlur(kernel_size=kernel_size)
                image = transform_blur(image)

        image_tensor = tf.to_tensor(image)
        target_tensor = torch.tensor(np.array(target))
        if self.args.input_space == 'GR':
            image_tensor_r = image_tensor[0].unsqueeze(0)
            image_tensor_grey = tf.to_tensor(tf.to_grayscale(image))
            image_tensor = torch.cat((image_tensor_r, image_tensor_grey), dim=0)
        if self.args.input_space == 'RGB':
            image_tensor = tf.normalize(image_tensor, mean=self.image_mean, std=self.image_std)
        if self.args.n_classes <= 2:
            target_tensor[target_tensor < 128] = 0
            target_tensor[target_tensor >= 128] = 1
        target_tensor = target_tensor.unsqueeze(0)
        return image_tensor, target_tensor

    def __getitem__(self, index):
        img_x_tr, img_y_tr = self.transform(self.img_x[index], self.img_y[index])
        return (img_x_tr, self.img_x_path[index]), (img_y_tr, self.img_y_path[index])

    def __len__(self):
        return len(self.img_x_path)

class Image2ImageLoader_zero_pad(Dataset):
    def __init__(self, x_path, y_path, mode, **kwargs):
        self.mode = mode
        self.args = kwargs['args']
        self.image_mean = [0.485, 0.456, 0.406]
        self.image_std = [0.229, 0.224, 0.225]

        x_img_name = sorted(filter(is_image, os.listdir(x_path)))
        y_img_name = sorted(filter(is_image, os.listdir(y_path)))

        self.img_x_path = [os.path.join(x_path, f) for f in x_img_name]
        self.img_y_path = [os.path.join(y_path, f) for f in y_img_name]

        assert len(self.img_x_path) == len(self.img_y_path), 'Images in directory must have same file indices!!'

        print(f'{utils.Colors.LIGHT_RED}Mounting data on memory...{self.__class__.__name__}:{self.mode}{utils.Colors.END}')
        self.img_x = [Image.open(p).convert('RGB') for p in self.img_x_path]
        self.img_y = [Image.open(p).convert('L') for p in self.img_y_path]

    def transform(self, image, target):
        resize_h = int(self.args.input_size[0])
        resize_w = int(self.args.input_size[1])
        image = tf.resize(image, [resize_h, resize_w])
        target = tf.resize(target, [resize_h, resize_w], interpolation=InterpolationMode.NEAREST)

        if self.mode != 'validation':
            random_gen = random.Random()
            if getattr(self.args, 'transform_hflip', False) and (random_gen.random() < 0.5):
                image = tf.hflip(image)
                target = tf.hflip(target)
            if getattr(self.args, 'transform_jitter', False) and (random_gen.random() < 0.8):
                transform_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
                image = transform_jitter(image)
            if getattr(self.args, 'transform_blur', False) and (random_gen.random() < 0.5):
                kernel_size = int((random.random() * 10 + 2.5).__round__())
                if kernel_size % 2 == 0: kernel_size -= 1
                transform_blur = transforms.GaussianBlur(kernel_size=kernel_size)
                image = transform_blur(image)

        image_tensor = tf.to_tensor(image)
        target_tensor = torch.tensor(np.array(target))
        if self.args.input_space == 'GR':
            image_tensor_r = image_tensor[0].unsqueeze(0)
            image_tensor_grey = tf.to_tensor(tf.to_grayscale(image))
            image_tensor = torch.cat((image_tensor_r, image_tensor_grey), dim=0)
        if self.args.input_space == 'RGB':
            image_tensor = tf.normalize(image_tensor, mean=self.image_mean, std=self.image_std)
        if self.args.n_classes <= 2:
            target_tensor[target_tensor < 128] = 0
            target_tensor[target_tensor >= 128] = 1
        target_tensor = target_tensor.unsqueeze(0)
        return image_tensor, target_tensor

    def __getitem__(self, index):
        img_x_tr, img_y_tr = self.transform(self.img_x[index], self.img_y[index])
        return (img_x_tr, self.img_x_path[index]), (img_y_tr, self.img_y_path[index])

    def __len__(self):
        return len(self.img_x_path)

class Image2ImageDataLoader_resize:
    def __init__(self, x_path, y_path, mode, batch_size=4, num_workers=0, pin_memory=True, **kwargs):
        g = torch.Generator()
        g.manual_seed(3407)
        self.image_loader = Image2ImageLoader_resize(x_path, y_path, mode=mode, **kwargs)
        self.Loader = MultiEpochsDataLoader(
            self.image_loader,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=(mode != 'validation'),
            worker_init_fn=seed_worker,
            generator=g,
            pin_memory=pin_memory
        )
    def __len__(self):
        return len(self.image_loader)

class Image2ImageDataLoader_zero_pad:
    def __init__(self, x_path, y_path, mode, batch_size=4, num_workers=0, pin_memory=True, **kwargs):
        g = torch.Generator()
        g.manual_seed(3407)
        self.image_loader = Image2ImageLoader_zero_pad(x_path, y_path, mode=mode, **kwargs)
        self.Loader = MultiEpochsDataLoader(
            self.image_loader,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=(mode != 'validation'),
            worker_init_fn=seed_worker,
            generator=g,
            pin_memory=pin_memory
        )
    def __len__(self):
        return len(self.image_loader)
