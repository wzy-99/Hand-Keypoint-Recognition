import os
import cv2
import json
import numpy as np
from tqdm import tqdm
from paddle.io import Dataset
from paddle.vision.transforms import Compose, Resize, Transpose, Normalize
from hotmap import hotmap
import config



transform = Compose([
    Resize(size=(config.INPUT_SIZE, config.INPUT_SIZE)),
    Normalize(mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], data_format='HWC'), # 标准化
    Transpose(), # 原始数据形状维度是HWC格式，经过Transpose，转换为CHW格式
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
        --root_path
            --1.jpg
            --1.json
            --2.jpg
            --2.json 
            ...... 
    """
    def __init__(self, root_path, class_number=1):
        self.root_path = root_path
        self.class_number = class_number
        self.sample = []
        for file_name in tqdm(os.listdir(self.root_path)):
            if file_name.endswith('.jpg'):
                name = file_name.split('.jpg')[0]
                json_name = name + '.json'
                json_path = os.path.join(self.root_path, json_name)
                image_path = os.path.join(self.root_path, file_name)
                if os.path.exists(json_path):
                    with open(json_path, 'r', encoding='UTF-8') as f:
                        json_obj = json.load(f)
                        hand_info = json_obj['info']
                        hand_num = len(hand_info)
                        labels = []
                        for hand in hand_info:
                            hand_parts = hand['pts']
                            # 只选取食指关节点
                            if '8' in hand_parts:
                                x = hand_parts['8']['x']
                                y = hand_parts['8']['y']
                                c = 0
                                labels.append([c, x, y])
                        if len(labels):
                            self.sample.append({'image_path': image_path, 'labels': labels})

    def __getitem__(self, idx):
        image = cv2.imread(self.sample[idx]['image_path'])
        h, w, c = image.shape
        label_image = np.zeros((self.class_number, config.LABEL_SIZE, config.LABEL_SIZE), dtype="float32")
        for lab in self.sample[idx]['labels']:
            c = lab[0]
            x = lab[1] * config.LABEL_SIZE / w
            y = lab[2] * config.LABEL_SIZE / h
            label_image[c, :, :] = hotmap(label_image[c, :, :], (x, y), config.RADIUS, (config.SIGMA, config.SIGMA))
        image = transform(image).astype("float32")
        ret = image, label_image
        return ret

    def __len__(self):
        return len(self.sample)


# class TrainDataset(Dataset):
#     """
#         --root_path
#           --json_name.json
#           --image_path
#             --0.jpg
#             --1.jpg
#             --2.jpg
#             ......
#     """
#     def __init__(self, root_path, json_name, class_number=1):
#         self.root_path = root_path
#         self.class_number = class_number
#         self.sample = []
        
#         json_path = os.path.join(self.root_path, json_name)
#         with open(json_path, 'r', encoding='UTF-8') as f:
#             json_obj = json.load(f)
#             root = json_obj['root']
#             for sample in root:
#                 img_path = sample['img_paths']
#                 image_path = os.path.join(self.root_path, img_path)
#                 img_width = sample['img_width']
#                 img_height = sample['img_height']
#                 joint_self = sample['joint_self']
#                 labels = []
#                 # 只选择食指关节点
#                 x = joint_self[8][0]
#                 y = joint_self[8][1]
#                 c = 0
#                 labels.append([c, x, y])
#                 # 剔除异常数据
#                 if x >=0 and x < img_width and y >= 0 and y < img_height:
#                     self.sample.append({'image_path': image_path, 'labels': labels})

#     def __getitem__(self, idx):
#         image = cv2.imread(self.sample[idx]['image_path'])
#         h, w, c = image.shape
#         label_image = np.zeros((self.class_number, config.LABEL_SIZE, config.LABEL_SIZE), dtype="float32")
#         for lab in self.sample[idx]['labels']:
#             c = lab[0]
#             x = lab[1] * config.LABEL_SIZE / w
#             y = lab[2] * config.LABEL_SIZE / h
#             label_image[c, :, :] = hotmap(label_image[c, :, :], (x, y), config.RADIUS, (config.SIGMA, config.SIGMA))
#         image = transform(image).astype("float32")
#         ret = image, label_image
#         return ret

#     def __len__(self):
#         return len(self.sample)