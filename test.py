import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

resnet = torchvision.models.resnet18(pretrained=True)
images = torch.randn(64, 3, 224, 224)
outputs = resnet(images)
print (outputs.size())