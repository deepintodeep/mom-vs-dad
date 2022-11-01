import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from pprint import pprint
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import os
import json
# import xmltodict # xml파일의 내용을 딕셔너리에 저장할 수 있는 메소드들이 들어있는 모듈입니다. 
from PIL import Image
import numpy as np
from tqdm import tqdm
class My_own_dataset():
    def __init__(self, root, train=True, transform=None, target_transform=None, resize=224) -> None:
        self.root = root
        self.train = train
        self.transform = transform
        self.resize_factor = resize
        self.dir_path = "./data/sample" # = self.root
        #self.family_ids = ["F0350", "F0351", "F0352"]
        self.family_ids = sorted(os.listdir( f'{self.dir_path}' )[1:])
        
        self.AorB = ( "A(친가)", "B(외가)" ) #paternal part, maternal part 
        # self.relations = ("1.Family", "2.Individuals", "3.Age")
        self.relations = ("2.Individuals") # individual만 사용할 것임.
        self.pathes = [ f"{self.dir_path}/{family_ids_}/{AorB_}/2.Individuals" \
            for family_ids_ in self.family_ids for AorB_ in self.AorB for relations_ in self.relations ]
        
        self.images = [] # element format: ("image.jpg", "json_file") or "image.jpg"
        self.json = []
        
        for file in self.pathes:
            inner_dir = os.listdir(f'{file}/')

            for file_name in inner_dir:
                if ( file_name[-4:]==".jpg" or file_name[-4:]=='.JPG' ):
                    # self.data.append( f"{file}/{file_name}" )
                    self.images.append( os.path.join(file, file_name) )
                else: self.json.append( os.path.join(file, file_name) )
            
                # (image, json) 이렇게 데이터에 넣을지 -> 정렬해서 넣어두자
        self.images.sort()
        self.json.sort()
        self.imgAndJson = list(zip(self.images, self.json))

        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index) -> tuple(list, str):
        # raw_img = ( Image.open(self.imgAndJson[index][0]).convert('RGB') ).resize((224,224)) # tuple json
        # print('name:', self.imgAndJson[index][0])
        raw_img = ( Image.open(self.imgAndJson[index][0]).convert('RGB') )
        
        with open(self.imgAndJson[index][1], "r") as st_json:
            st_python = json.load(st_json)
            family_id = st_python["family_id"]
            bounding = st_python['member'][0]['regions'][0]['boundingbox']
        
        # transform func
        raw_img_transform = transforms.Compose( [transforms.PILToTensor(), transforms.Resize((256,256))] )
        
        # croping start
        crop_list = []
        for idx in range(7):
            x = int(bounding[idx]['x']) # 각 bounding box에서의 좌표들
            y = int(bounding[idx]['y'])
            w = int(bounding[idx]['w'])
            h = int(bounding[idx]['h'])
            crop = raw_img.crop((x,y,x+w,y+h))
            # crop.show() # for testing.
            crop = torch.divide( raw_img_transform(crop), 255)
            crop_list.append(crop)
            
        return crop_list, family_id # tuple([7 of tensors], family_id)
        # if self.transform is not None:
        #     img = self.transform(img)

        
    def json_parsing(self, jsondir):
        with open(f"{jsondir}", "r") as st_json:
            st_python = json.load(st_json)
        json_length = len(st_python['member'])
        for idx in range(json_length):
            print( st_python['member'][idx]['regions'][0]['boundingbox'] )
            # print(type(st_python['member'][idx]))
        # print(st_python["family_id"], end="\t")
        # print(st_python["member"]["boundingbox"])
        
if __name__ == '__main__':
    instance = My_own_dataset("../data/sample/", transform=transforms.ToTensor())
    print(instance.__getitem__(0))
    # print(instance.__getitem__(0).size()) # return 넣으면 풀기
    print('instance len:',instance.__len__())
    
    