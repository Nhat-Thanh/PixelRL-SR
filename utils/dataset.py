from utils.common import *
import numpy as np
import torch
import os

class dataset:
    def __init__(self, dataset_dir, subset):
        self.dataset_dir = dataset_dir
        self.subset = subset

        self.bicubic = torch.Tensor([])
        self.bicubic_file = os.path.join(self.dataset_dir, f"bicubic_{self.subset}.npy")
        self.lr = torch.Tensor([])
        self.lr_file = os.path.join(self.dataset_dir, f"data_{self.subset}.npy")
        self.hr = torch.Tensor([])
        self.hr_file = os.path.join(self.dataset_dir, f"labels_{self.subset}.npy")
        self.cur_idx = 0

    def generate(self, lr_crop_size, hr_crop_size, transform=False):      
        if exists(self.bicubic_file) and exists(self.lr_file) and exists(self.hr_file):
            print(f"{self.bicubic_file}, {self.lr_file} and {self.hr_file} HAVE ALREADY EXISTED\n")
            return
        bicucbic = []
        lr = []
        hr = []
        step = hr_crop_size - 1

        subset_dir = os.path.join(self.dataset_dir, self.subset)
        ls_images = sorted_list(subset_dir)
        scale = hr_crop_size // lr_crop_size
        # if scale == 2, assign it to 3, else assign it to 4
        # scale = 3 * (scale == 2) + 4 * (scale > 2)
        for image_path in ls_images:
            print(image_path)
            hr_image = read_image(image_path)

            h = hr_image.shape[1]
            w = hr_image.shape[2]
            for x in np.arange(start=0, stop=h-hr_crop_size, step=step):
                for y in np.arange(start=0, stop=w-hr_crop_size, step=step):
                    subim_hr  = hr_image[:, x : x + hr_crop_size, y : y + hr_crop_size]
                    if transform:
                        subim_hr = random_transform(subim_hr)

                    subim_lr = gaussian_blur(subim_hr, sigma=0.55)
                    subim_bicucbic = make_lr(subim_lr, scale)
                    subim_lr = resize_bicubic(subim_lr, lr_crop_size, lr_crop_size)

                    subim_bicucbic = rgb2ycbcr(subim_bicucbic)
                    subim_lr = rgb2ycbcr(subim_lr)
                    subim_hr = rgb2ycbcr(subim_hr)

                    subim_bicucbic = norm01(subim_bicucbic)
                    subim_lr = norm01(subim_lr)
                    subim_hr = norm01(subim_hr)

                    bicucbic.append(subim_bicucbic.numpy())
                    lr.append(subim_lr.numpy())
                    hr.append(subim_hr.numpy())

        bicucbic = np.array(bicucbic)
        lr = np.array(lr)
        hr = np.array(hr)

        np.save(self.bicubic_file, bicucbic)
        np.save(self.lr_file, lr)
        np.save(self.hr_file, hr)

    def load_data(self, shuffle_arrays=True):
        if not exists(self.lr_file):
            raise ValueError(f"\n{self.lr_file} and {self.hr_file} DO NOT EXIST\n")
        self.bicubic = np.load(self.bicubic_file)
        self.lr = np.load(self.lr_file)
        self.hr = np.load(self.hr_file)

        if shuffle_arrays:
            indices = np.arange(0, self.lr.shape[0], 1)
            np.random.shuffle(indices)
            self.bicubic = self.bicubic[indices]
            self.lr = self.lr[indices]
            self.hr = self.hr[indices]

        self.bicubic = torch.as_tensor(self.bicubic)
        self.lr = torch.as_tensor(self.lr)
        self.hr = torch.as_tensor(self.hr)

    def get_batch(self, batch_size, shuffle_each_epoch=True):
        # Ignore remaining dataset because of  
        # shape error when run torch.mean()
        isEnd = False
        if self.cur_idx + batch_size > self.lr.shape[0]:
            isEnd = True
            self.cur_idx = 0
            if shuffle_each_epoch:
                indices = torch.randperm(self.lr.shape[0])
                self.bicubic = torch.index_select(self.bicubic, 0, indices)
                self.lr = torch.index_select(self.lr, 0, indices)
                self.hr = torch.index_select(self.hr, 0, indices)

        bicubic = self.bicubic[self.cur_idx : self.cur_idx + batch_size]
        lr = self.lr[self.cur_idx : self.cur_idx + batch_size]
        hr = self.hr[self.cur_idx : self.cur_idx + batch_size]
        self.cur_idx += batch_size

        return bicubic, lr, hr, isEnd
