import os
import cv2
import json
import numpy as np
from tqdm import tqdm
from paddle.io import Dataset
from paddle.vision.transforms import Compose, Resize, Transpose, Normalize, ColorJitter
from hotmap import hotmap
import config


transform = Compose([
    Resize(size=(config.INPUT_SIZE, config.INPUT_SIZE)),
    ColorJitter(brightness=0.1, contrast=0.3, saturation=0.3, hue=0.1),
    Normalize(mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], data_format='HWC'),
    Transpose(),
])


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


class TrainDataset_v1(Dataset):
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
            dat = image[y0:y1, x0:x1, :]
            label_image = np.zeros((config.CLASS_NUMBER, config.LABEL_SIZE, config.LABEL_SIZE), dtype="float32")
            for lab in config.USE_LABEL:
                c = config.LABEL2ID[lab]
                x = (points[lab]['x'] - x0) * config.LABEL_SIZE / w 
                y = (points[lab]['y'] - y0 + config.DY) * config.LABEL_SIZE / h 
                label_image[c, :, :] = hotmap(label_image[c, :, :], (x, y), config.RADIUS, (config.SIGMA, config.SIGMA))
            # watch = cv2.resize(label_image[0], (w, h)) 
        else: # 不裁剪
            dat = image
            label_image = np.zeros((config.CLASS_NUMBER, config.LABEL_SIZE, config.LABEL_SIZE), dtype="float32")
            for lab in config.USE_LABEL:
                c = config.LABEL2ID[lab]
                x = points[lab]['x'] * config.LABEL_SIZE / width 
                y = (points[lab]['y'] + config.DY) * config.LABEL_SIZE / height 
                label_image[c, :, :] = hotmap(label_image[c, :, :], (x, y), config.RADIUS, (config.SIGMA, config.SIGMA))
        dat = transform(dat).astype("float32")
        ret = dat, label_image
        return ret

    def __len__(self):
        return len(self.sample)


class TrainDataset_v2(Dataset):
    """
        2维坐标标签
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
                        hand_info = json_obj['hand_info']
                        for hand in hand_info:
                            hand_parts = hand['hand_parts']
                            location = hand['location']
                            width = location['width']
                            height = location['height']
                            left = location['left']
                            top = location['top']
                            self.sample.append({'image_path': image_path, 'points': hand_parts, 'location': [left, top, width, height]})

    def __getitem__(self, idx):
        image_path = self.sample[idx]['image_path']
        points = self.sample[idx]['points']
        x0, y0, w, h = self.sample[idx]['location']
        image = cv2.imread(image_path)
        height, width, channel = image.shape
        x1 = min(x0 + config.MARGIN + w, width - 1)
        y1 = min(y0 + config.MARGIN + h, height - 1)
        x0 = max(x0 - config.MARGIN, 0)
        y0 = max(y0 - config.MARGIN, 0)
        w = x1 - x0
        h = y1 - y0
        dat = image[y0:y1, x0:x1, :]
        label_array = np.zeros(shape=(config.CLASS_NUMBER, 2), dtype='float32')
        for lab in config.USE_LABEL:
            c = config.LABEL2ID[lab]
            x = (points[lab]['x'] - x0) * config.LABEL_SIZE / w
            y = (points[lab]['y'] - y0 + config.DY) * config.LABEL_SIZE / h
            label_array[c, 0] = x
            label_array[c, 1] = y
        dat = transform(dat).astype("float32")
        ret = dat, label_array
        return ret

    def __len__(self):
        return len(self.sample)