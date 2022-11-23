import torch
import torchvision.transforms as transforms

import os
import json
from PIL import Image

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

        # Image Path List
        image_path_list = sorted(os.listdir(os.path.join(self.root, 'image')))
        image_path_list = [os.path.join(self.root, 'image', path) for path in image_path_list]
        image_path_list = image_path_list[1:]

        self.image_path_list = []

        for image_path in image_path_list:
            for aorb in AorB:
                path = os.path.join(image_path, aorb, '2.Individuals')
                path_list = os.listdir(path)
                path_list = [os.path.join(path, file) for file in path_list]
                self.image_path_list += path_list
        self.image_path_list = sorted(self.image_path_list)

        # Json Path List
        json_path_list = sorted(os.listdir(os.path.join(self.root, 'label')))
        json_path_list = [os.path.join(self.root, 'label', path) for path in json_path_list]
        json_path_list = json_path_list[1:]

        self.json_path_list = []

        for json_path in json_path_list:
            for aorb in AorB:
                path = os.path.join(json_path, aorb, '2.Individuals')
                path_list = os.listdir(path)
                path_list = [os.path.join(path, file) for file in path_list]
                self.json_path_list += path_list
        self.json_path_list = sorted(self.json_path_list)
        
    def __len__(self):
        return len(self.image_path_list)
    
    def __getitem__(self, idx):
        image_path = self.image_path_list[idx]
        json_path = self.json_path_list[idx]

        raw_image = Image.open(image_path)

        with open(json_path, 'r') as f:
            label = json.load(f)
            family_id = label['family_id']
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
        return images, family_id

dataset = SimilarityDataset()
inputs, labels = dataset[0]
print(inputs.shape)
print(labels)

# [2, 7, 3, 256, 256]