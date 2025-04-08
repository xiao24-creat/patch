# sys
import os
import sys
import numpy as np
import random
import pickle
import cv2  # 新增导入，用于处理图像

# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms

# visualization
import time

# operation
from . import tools

class Feeder(torch.utils.data.Dataset):
    """ Feeder for skeleton-based action recognition
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
        label_path: the path to label
        image_paths: the list of image paths corresponding to each sample
        random_choose: If true, randomly choose a portion of the input sequence
        random_shift: If true, randomly pad zeros at the begining or end of sequence
        window_size: The length of the output sequence
        normalization: If true, normalize input sequence
        debug: If true, only use the first 100 samples
    """

    def __init__(self,
                 data_path,
                 label_path,
                 image_paths,  # 新增参数，用于存储图像路径列表
                 random_choose=False,
                 random_move=False,
                 window_size=-1,
                 debug=False,
                 mmap=True):
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.image_paths = image_paths  # 新增属性，存储图像路径列表
        self.random_choose = random_choose
        self.random_move = random_move
        self.window_size = window_size

        self.load_data(mmap)

    def load_data(self, mmap):
        # data: N C V T M

        # load label
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)

        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]
            self.image_paths = self.image_paths[0:100]  # 新增，确保图像路径也进行裁剪

        self.N, self.C, self.T, self.V, self.M = self.data.shape

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        data_numpy = np.array(self.data[index])
        label = self.label[index]
        image_path = self.image_paths[index]  # 获取当前样本的图像路径

        patch_features = []
        for t in range(data_numpy.shape[1]):  # 遍历时间步
            frame_patch_features = []
            for v in range(data_numpy.shape[2]):  # 遍历节点
                x, y = data_numpy[0, t, v], data_numpy[1, t, v]
                patch = self.extract_patch_features(image_path, (x, y))
                frame_patch_features.append(patch)
            patch_features.append(frame_patch_features)
        patch_features = np.array(patch_features)

        # processing
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)

        return data_numpy, patch_features, label

    def extract_patch_features(self, image_path, point):
        image = cv2.imread(image_path)
        x, y = point
        h, w, _ = image.shape
        patch_size = 16  # 假设 patch 大小为 16x16
        x_min = max(0, int(x - patch_size // 2))
        x_max = min(w, int(x + patch_size // 2))
        y_min = max(0, int(y - patch_size // 2))
        y_max = min(h, int(y + patch_size // 2))
        patch = image[y_min:y_max, x_min:x_max]
        patch = cv2.resize(patch, (patch_size, patch_size))
        patch = patch.flatten()  # 展平为一维向量
        return patch

    def extract_patch_features(self, image_path, point, patch_size=16):
        try:
            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to read image from {image_path}")
            # 获取图像的高度和宽度
            h, w, _ = image.shape
            # 计算 patch 的边界
            x, y = point
            x_min = max(0, int(x - patch_size // 2))
            x_max = min(w, int(x + patch_size // 2))
            y_min = max(0, int(y - patch_size // 2))
            y_max = min(h, int(y + patch_size // 2))
            # 提取 patch
            patch = image[y_min:y_max, x_min:x_max]
            # 调整 patch 的大小
            patch = cv2.resize(patch, (patch_size, patch_size))
            # 归一化处理
            patch = patch / 255.0
            # 展平为一维向量
            patch = patch.flatten()
            return patch
        except Exception as e:
            print(f"Error extracting patch features: {e}")
            return np.zeros(patch_size * patch_size * 3)  # 返回全零向量作为默认值