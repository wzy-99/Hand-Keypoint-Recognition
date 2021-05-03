import os
import cv2
import json
import numpy as np
from tqdm import tqdm
import paddle
from paddle.io import Dataset
from paddle.vision.transforms import Compose, Resize, Transpose, Normalize, ColorJitter
from hotmap import hotmap
import config
import random


transform = Compose([
    Resize(size=(config.INPUT_SIZE, config.INPUT_SIZE)),
    ColorJitter(brightness=0.1, contrast=0.3, saturation=0.3, hue=0.1),
    Normalize(mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], data_format='HWC'),
    Transpose(),
])


def random_region(width, height, box):
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0

    w_min_ratio = max(config.W_MIN_RATIO, bw / width)
    if w_min_ratio > config.W_MAX_RATIO:
        print('MAX_RATIO is too low')
        w_max_ratio = 1.0
    else:
        w_max_ratio = config.W_MAX_RATIO
    h_min_ratio = max(config.H_MIN_RATIO, bh / height)
    if w_min_ratio > config.H_MAX_RATIO:
        print('MAX_RATIO is too low')
        w_max_ratio = 1.0
    else:
        h_max_ratio = config.H_MAX_RATIO

    w_ratio = random.random() * (w_max_ratio - w_min_ratio) + w_min_ratio
    h_ratio = random.random() * (h_max_ratio - h_min_ratio) + h_min_ratio

    w = int(width * w_ratio)
    h = int(height * h_ratio)

    lmin = max(0, x1 - w)
    lmax = min(width - 1 - w, x0)
    tmin = max(0, y1 - h)
    tmax = min(height -1 - h, y0)

    left = random.randint(lmin, lmax)
    top = random.randint(tmin, tmax)
    
    return left, top, w, h


class TestDataset(Dataset):
    def __init__(self, image_path):
        self.image_path = image_path
        self.sample = []
        self.indexs = []

        for file_name in os.listdir(self.image_path):
            if file_name.endswith('.jpg'):
                self.sample.append(os.path.join(self.image_path, file_name))
        
    def __getitem__(self, idx):
        self.indexs.append(self.sample[idx])
        image = cv2.imread(self.sample[idx])
        image = transform(image).astype("float32")
        ret = image
        return ret

    def __len__(self):
        return len(self.sample)


class TrainDataset(Dataset):
    """
        hotmap标签
        --root_path
            --1.jpg
            --1.json
            --2.jpg
            --2.json 
            ...... 
    """
    def __init__(self, root_path):
        self.root_path = root_path
        self.sample = []
        for image_name in tqdm(os.listdir(self.root_path)):
            if image_name.endswith('.jpg'):
                name = image_name.split('.jpg')[0]
                json_name = name + '.json'
                json_path = os.path.join(self.root_path, json_name)
                image_path = os.path.join(self.root_path, image_name)
                if os.path.exists(json_path):
                    with open(json_path, 'r', encoding='UTF-8') as f:
                        json_obj = json.load(f)
                        if 'hand_info' in json_obj:
                            hand_info = json_obj['hand_info']
                            for hand in hand_info:
                                hand_parts = hand['hand_parts']
                                location = hand['location']
                                width = location['width']
                                height = location['height']
                                left = location['left']
                                top = location['top']
                                self.sample.append({'image_path': image_path, 'points': hand_parts, 'location': [left, top, width, height]})
                        elif 'info' in json_obj:
                            info = json_obj['info']
                            for hand in info:
                                pts = hand['pts']
                                self.sample.append({'image_path': image_path, 'points': pts})

    def __getitem__(self, idx):
        image_path = self.sample[idx]['image_path']
        points = self.sample[idx]['points']
        image = cv2.imread(image_path)
        height, width, channel = image.shape
        if 'location' in self.sample[idx]: # 进行裁剪
            x0, y0, w, h = self.sample[idx]['location']
            x1 = min(x0 + config.MARGIN + w, width - 1)
            y1 = min(y0 + config.MARGIN + h, height - 1)
            x0 = max(x0 - config.MARGIN, 0)
            y0 = max(y0 - config.MARGIN, 0)
            w = x1 - x0
            h = y1 - y0
            nx0, ny0, nw, nh = random_region(width, height, [x0, y0, x1, y1])
            nx1 = nx0 + nw
            ny1 = ny0 + nh
            dat = image[ny0:ny1, nx0:nx1, :]
            label_image = np.zeros((config.CLASS_NUMBER, config.LABEL_SIZE, config.LABEL_SIZE), dtype="float32")
            for lab in config.USE_LABEL:
                c = config.LABEL2ID[lab]
                cx = (points[lab]['x'] - nx0) * config.LABEL_SIZE / nw 
                cy = (points[lab]['y'] - ny0 + config.DY) * config.LABEL_SIZE / nh 
                label_image[c, :, :] = hotmap(label_image[c, :, :], (cx, cy), config.RADIUS, (config.SIGMA, config.SIGMA))
        else: # 不裁剪
            dat = image
            label_image = np.zeros((config.CLASS_NUMBER, config.LABEL_SIZE, config.LABEL_SIZE), dtype="float32")
            for lab in config.USE_LABEL:
                c = config.LABEL2ID[lab]
                cx = points[lab]['x'] * config.LABEL_SIZE / width 
                cy = (points[lab]['y'] + config.DY) * config.LABEL_SIZE / height 
                label_image[c, :, :] = hotmap(label_image[c, :, :], (cx, cy), config.RADIUS, (config.SIGMA, config.SIGMA))
        dat = transform(dat).astype("float32")
        ret = dat, label_image
        return ret

    def __len__(self):
        return len(self.sample)


if __name__ == '__main__':
    paddle.seed(1)
    random.seed(1)
    ds = TrainDataset_v1('mydata')
    for x, label in ds:
        cv2.imshow('x', x[1, :, :])
        cv2.imshow('l', label[0, :, :])
        cv2.waitKey(0)