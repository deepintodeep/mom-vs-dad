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
        # torchvision/data/sample/F0350/A(친가)/1.Family
        self.dir_path = "../data/sample/" # = self.root
        #self.family_ids = ["F0350", "F0351", "F0352"]
        self.family_ids = os.listdir( f'{self.dir_path}' )[1:]
        # print(self.family_ids)
        self.AorB = ( "A(친가)", "B(외가)" ) #paternal part, maternal part 
        self.relations = ("1.Family", "2.Individuals", "3.Age")
        self.pathes = [ f"{self.dir_path}{family_ids_}/{AorB_}/{relations_}" \
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
        # print(len(self.imgAndJson))
        # print(len(self.imgAndJson[0]))
        # pprint(self.imgAndJson)

        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        # 나중엔 바운딩 박스 모양으로 사진 잘라야됨 
        # return -> 총 image tensor 7개, family_id 1개 있게됨.
        raw_img = ( Image.open(self.imgAndJson[index][0]).convert('RGB') ).resize((224,224)) # tuple json
        # print(self.images[index])
        print(self.imgAndJson[index][1])
        with open(self.imgAndJson[index][1], "r") as st_json:
            st_python = json.load(st_json)
            json_length = len(st_python['member'])
            for idx in range(json_length):
                # print( st_python['member'][idx]['regions'][0]['boundingbox'] )
                pass
        
        raw_img.show() 
        raw_img_transform = transforms.Compose( [transforms.PILToTensor(), transforms.Resize((224,224))] )
        raw_img = torch.divide( raw_img_transform(raw_img), 255 )
        
        # if self.transform is not None:
        #     img = self.transform(img)
        
        # return ( 7 of tensors, family_id )
        
    def json_parsing():
        pass
if __name__ == '__main__':
    instance = My_own_dataset("../data/sample/", transform=transforms.ToTensor())
    instance.__getitem__(0)
    # print(instance.__getitem__(0).size()) # return 넣으면 풀기
    print(instance.__len__())
    