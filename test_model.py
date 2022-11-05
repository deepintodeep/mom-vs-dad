import torch
from torchsummary import summary

import torchvision.models.resnet as resnet
import torch.nn as nn
import torch.optim as optim
from torchvision import models

conv1x1 = resnet.conv1x1
Bottleneck = resnet.Bottleneck
BasicBlock = resnet.BasicBlock

class ResNet(nn.Module):
  def __init__(self, block, layers, num_classes = 1000, zero_init_residual = True):
    super(ResNet, self).__init__()
    self.inplanes = 32  #inplanes: 64에서 32로 변경

    self.conv1 = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size = 7, stride = 2, padding = 3, bias = False),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

    self.layer1 = self._make_layer(block, 32, layers[0], stride = 1)
    self.layer2 = self._make_layer(block, 64, layers[1], stride = 2)
    self.layer3 = self._make_layer(block, 128, layers[2], stride = 2)
    self.layer4 = self._make_layer(block, 256, layers[3], stride = 2)

    self.avgpool = nn.AdaptiveAvgPool2d((1,1)) 
    self.fc = nn.Linear(256*block.expansion, num_classes)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity = 'relu')
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias,0)

    if zero_init_residual:
      for m in self.modules():
        if isinstance(m, Bottleneck):
          nn.init.constant_(m.bn3.weight, 0)
        elif isinstance(m, BasicBlock):
          nn.init.constant_(m.bn2.weight, 0)

  def _make_layer(self, block, planes, blocks, stride = 1):
    downsample = None
    if stride !=1 or self.inplanes !=planes * block.expansion:
      downsample = nn.Sequential(
          conv1x1(self.inplanes, planes * block.expansion, stride),
          nn.BatchNorm2d(planes*block.expansion),
      )

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample))
    self.inplanes = planes * block.expansion
    for _ in range (1,blocks):
      layers.append(block(self.inplanes, planes))

    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv1(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)

    return x 

def resnet50():
  return ResNet(Bottleneck, [3, 4, 6, 3])


#With pretrained model
class test_Model(nn.Module):
  def __init__(self):
    super().__init__()
    self.pretrained_model = models.resnet50(weights = "IMAGENET1K_V1") #load pretrained model
    self.pretrained = nn.Sequential(*(list(self.pretrained_model.children())[:-1]))
    self.extra = nn.Conv2d(2048, 1024, kernel_size = 3, stride = 1, padding = 1, bias = False) #[2048,1,1] --> [1024, 1, 1]
    

  def forward(self,x):
    x = self.pretrained(x)
    x = self.extra(x)
    return x

def resnet50_pt():
  return test_Model()




if __name__ == '__main__':
  device = 'cuda' if torch.cuda.is_available() else 'cpu'

  #model without pretrained weights
  model = resnet50().to(device)
  summary(model, input_size = (3,256,256), device = device)

  print("\n\n")

  #model with pretrained weights
  new_model = resnet50_pt().to(device)
  summary(new_model, input_size = (3,256, 256), device = device)