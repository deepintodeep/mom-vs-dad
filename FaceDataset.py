"""
FaceDataset.py
"""

import os
import torch
from PIL import Image
import json
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class FaceDataset(torch.utils.data.Dataset):
    def __init__(self, data_folder, transforms=None):
        self.transforms = transforms
        self.data_folder = data_folder
        # |___ data
        #      |___ train
        #           |___ images
        #                |___ FOO01_IND_D_18_0_01.jpg
        #                |___ FOO01_IND_D_18_0_02.jpg
        #                ...
        #           |___ labels
        #                |___ FOO01_IND_D_18_0_01.json
        #                |___ FOO01_IND_D_18_0_02.json
        #                ...
        #      |___ validation
        #           ...

        #전체 파일 목록 불러오고, img와 annot(json)으로 나눠 경로 저장
        self.root_imgs = os.path.join(data_folder, "images")
        self.root_annots = os.path.join(data_folder, "labels")
        self.list_imgs = list(sorted(os.listdir(self.root_imgs)))
        self.list_annots = list(sorted(os.listdir(self.root_annots)))

    def _bboxCorTransform(self, x, y, w, h):
        x0, x1, y0, y1 = x, x+w, y, y+h  # x, y가 중심 좌표가 아니라 좌상단 지점 좌표
        return [x0, y0, x1, y1]


    def __getitem__(self, idx):
        # 이미지, json 경로
        path_imgs = os.path.join(self.root_imgs, self.list_imgs[idx])
        path_annots = os.path.join(self.root_annots, self.list_annots[idx])

        # 이미지 불러오기
        img = Image.open(path_imgs).convert("RGB")

        # annotaion 불러오기
        f = open(path_annots, encoding="UTF-8")
        annot = json.loads(f.read())

        # boundingbox 목록, {idx, x, y, w, h}
        bounding_boxes = annot["member"][0]["regions"][0]["boundingbox"]

        # boundingbox 개수
        num_objs = len(bounding_boxes)

        # boundingbox, lable 추가
        boxes = []
        labels = []
        for i in bounding_boxes:
            boxes.append(self._bboxCorTransform(*list(map(int, [i["x"], i["y"], i["w"], i["h"]]))))
            labels.append(int(i["idx"]) + 1)

        image_id = torch.tensor([idx])

        # target transform
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

        # pytorch tutorial에 있는 finetuning 코드로 확인해보기 위해 target 추가 설정
        target["image_id"] = image_id
        target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target["iscrowd"] = torch.zeros((num_objs,), dtype=torch.int64)

        # image transform
        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        # 데이터셋 크기 조절
        #return len(self.list_imgs)
        return 2000

# collate_fn
def collate_fn(batch):
    return tuple(zip(*batch))

# image와 bounding box 확인용
def plot_image(img, annotation):
    fig, ax = plt.subplots(1)
    plt.imshow(img.permute(1, 2, 0))
    annotation = annotation[0]

    for idx in range(len(annotation["boxes"])):
        xmin, ymin, xmax, ymax = annotation["boxes"][idx]
        if annotation['labels'][idx] == 0:
            rect = patches.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), linewidth=1, edgecolor='r',
                                     facecolor='none')
        elif annotation['labels'][idx] == 1:
            rect = patches.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), linewidth=1, edgecolor='g',
                                     facecolor='none')
        else:
            rect = patches.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), linewidth=1, edgecolor='orange',
                                     facecolor='none')
        ax.add_patch(rect)
    plt.show()

if __name__ == '__main__':
    dir = os.path.join(os.getcwd(), "data", "train")
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    train_dataset = FaceDataset(dir, transform)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, collate_fn=collate_fn)

    for image, labels in train_dataloader:
        plot_image(image[0], labels)