"""
FinetuningTest.py
데이터셋 확인용 파이토치 튜토리얼 수정 코드
1) maskRCNN -> FasterRCNN으로 모델 변경
2) class 수 7 + 1개로 수정
3) 학습 시 evaluation 과정 생략 (1.train)
4) 1800개 이미지로 학습한 모델 불러와서 임의의 이미지에 적용 가능 (2.load)
"""

import torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from FaceDataset import FaceDataset
import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from engine import train_one_epoch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def plot_image(img, annotation):
    fig, ax = plt.subplots(1)
    plt.imshow(img)
    annotation = annotation[0]

    checked = [0 for i in range(8)]
    for idx in range(len(annotation["boxes"])):
        if checked[annotation["labels"][idx]]:
            continue
        else:
            checked[annotation["labels"][idx]] = 1
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


def collate_fn(batch):
    return tuple(zip(*batch))


def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weight=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def train():
    # hyperparameters
    batch_size = 1
    num_epochs = 1
    num_classes = 8

    ## optimizer
    lr = 0.005
    momentum = 0.9
    weight_decay = 0.0005

    ## lr scheduler
    step_size = 3
    gamma = 0.1

    # transform
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # directory path
    dir_train = os.path.join(os.getcwd(), "data", "train")
    dir_test = os.path.join(os.getcwd(), "data", "validation")

    # dataset
    dataset = FaceDataset(dir_train, transform)
    dataset_test = FaceDataset(dir_test, transform)

    # dataloader
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=4,
        collate_fn=collate_fn)

    # model
    model = get_model(num_classes=num_classes)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    # set optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr,
                                momentum=momentum, weight_decay=weight_decay)

    # set lr scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=step_size,
                                                   gamma=gamma)

    # train
    for epoch in range(num_epochs):
        # 10회 마다 출력
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # lr update
        lr_scheduler.step()

    torch.save(model.state_dict(), 'model_weights.pth')


def load():
    model = get_model(num_classes=8)

    dir_test = os.path.join(os.getcwd(), "data", "validation")
    root_imgs = os.path.join(dir_test, "images")
    list_imgs = list(sorted(os.listdir(root_imgs)))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.load_state_dict(torch.load('model_weights.pth'))
    model.eval()

    for idx in range(len(list_imgs)):
        path_imgs = os.path.join(root_imgs, list_imgs[idx])
        image = Image.open(path_imgs).convert("RGB")

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        image = transform(image)
        imgforinput = image.view([1, *image.shape]).to(device)

        a = model.forward(images=imgforinput)
        for i in range(len(a)):
            for j in a[i].keys():
                a[i][j] = a[i][j].detach().to("cpu")

        plot_image(image[0], a)


if __name__ == '__main__':
    a = int(input(("1. train | 2. load\n")))
    if a == 1:
        train()
    load()