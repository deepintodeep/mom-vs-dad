import torch
import torchvision.transforms as transforms

import os
import json
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

class SimilarityDataset:
    def __init__(self, root='./data', train=True):
        self.root = root
        self.train = train
        
        if self.train:
            self.root = os.path.join(self.root, 'train')
        else:
            self.root = os.path.join(self.root, 'test')

        AorB = ['A(친가)', 'B(외가)']

        self.transform = transforms.Compose([
            transforms.PILToTensor(),
            transforms.Resize((256, 256))
        ])

        # 가족 디렉터리 경로
        family_image_dir_paths = sorted(os.listdir(os.path.join(self.root, 'image')))
        family_image_dir_paths = [os.path.join(self.root, 'image', path) for path in family_image_dir_paths]
        family_image_dir_paths = family_image_dir_paths[1:]
        # print(f'a: {family_image_dir_paths[:3]}')

        self.family_image_path_dict = {}

        for dir_path in family_image_dir_paths:
            family_name = dir_path.split('/')[-1]
            self.family_image_path_dict[family_name] = []
            for aorb in AorB:
                path = os.path.join(dir_path, aorb, '2.Individuals')
                path_list = os.listdir(path)
                path_list = [os.path.join(path, file) for file in path_list]
                self.family_image_path_dict[family_name] += path_list
            self.family_image_path_dict[family_name] = sorted(self.family_image_path_dict[family_name])
        # print(f'b: {len(self.family_image_path_dict["TS0001"])}')
        # print(f'c: {self.family_image_path_dict["TS0001"][:3]}')
        
        # Json Path List
        family_json_dir_paths = sorted(os.listdir(os.path.join(self.root, 'label')))
        family_json_dir_paths = [os.path.join(self.root, 'label', path) for path in family_json_dir_paths]
        family_json_dir_paths = family_json_dir_paths[1:]
        # print(f'd: {family_json_dir_paths[:3]}')

        self.family_json_path_dict = {}

        for dir_path in family_json_dir_paths:
            family_name = dir_path.split('/')[-1]
            self.family_json_path_dict[family_name] = []
            for aorb in AorB:
                path = os.path.join(dir_path, aorb, '2.Individuals')
                path_list = os.listdir(path)
                path_list = [os.path.join(path, file) for file in path_list]
                self.family_json_path_dict[family_name] += path_list
            self.family_json_path_dict[family_name] = sorted(self.family_json_path_dict[family_name])
        # print(f'e: {len(self.family_json_path_dict["TL0001"])}')
        # print(f'f: {self.family_json_path_dict["TL0001"][:3]}')

        self.image_pair_list = self.make_pair(self.family_image_path_dict)
        self.json_pair_list = self.make_pair(self.family_json_path_dict)
        
    def make_pair(self, data_dict):
        pair_list=[]
        for key in data_dict.keys():
            for i in range(32):  
                for j in range(32):
                    a=data_dict[key][i]
                    b=data_dict[key][j+32]
                    pair_list.append((a,b))
            for i in range(32):  
                for j in range(32):
                    a=data_dict[key][i]
                    b=data_dict[key][j+64]
                    pair_list.append((a,b))

            for i in range(32):  
                for j in range(32):
                    a=data_dict[key][i+32]
                    b=data_dict[key][j+64]
                    pair_list.append((a,b))
        return pair_list
    
    def __len__(self):
        return len(self.image_pair_list)
    
    def __getitem__(self, idx):
        image_path_0, image_path_1 = self.image_pair_list[idx]
        json_path_0, json_path_1 = self.json_pair_list[idx]

        image_0 = self.get_image(image_path_0, json_path_0)
        image_1 = self.get_image(image_path_1, json_path_1)
        
        return image_0, image_1


    def get_image(self, image_path, json_path):
        raw_image = Image.open(image_path)

        with open(json_path, 'r') as f:
            label = json.load(f)
            bounding_boxes = label['member'][0]['regions'][0]['boundingbox']
        
        images = []
        for bounding_box in bounding_boxes:
            x = int(bounding_box['x'])
            y = int(bounding_box['y'])
            w = int(bounding_box['w'])
            h = int(bounding_box['h'])

            image = raw_image.crop((x, y, x + w, y + h))
            image = self.transform(image)
            image = torch.divide(image, 255)
            images.append(image)
        images = torch.cat(images, dim=0)
        images = images.reshape(-1, 3, 256, 256)
        return images

# dataset = SimilarityDataset()

# image_0, image_1 = dataset[0]
# print(image_0.shape)
# print(image_1.shape)
# [2, 7, 3, 256, 256]