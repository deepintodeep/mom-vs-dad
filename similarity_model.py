import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class SimilarityNet(nn.Module):
    def __init__(self, n_images=7):
        super().__init__()
        self.n_images = n_images
        self.networks = nn.ModuleList()

        for _ in range(n_images):
            network = resnet18(ResNet18_Weights.IMAGENET1K_V1)
            network.fc = nn.Linear(512, 128)
            self.networks.append(network)
        
    def forward(self, inputs):
        outputs = []
        for i in range(self.n_images):
            output = self.networks[i](inputs[:, i])
            outputs.append(output)
        outputs = torch.cat(outputs, dim=1)
        return outputs

# net = resnet18(ResNet18_Weights.IMAGENET1K_V1)
# net.fc = nn.Linear(512, 128)
# inputs = torch.ones(4, 7, 3, 256, 256)
# input = inputs[:, 0]
# print(input.shape)
# outputs = net(input)
# print(outputs.shape)

net = SimilarityNet()

inputs = torch.ones(4, 7, 3, 256, 256)
outputs = net(inputs)
print(outputs.shape)