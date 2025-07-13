import torch
import torchvision

class Resnet18(torch.nn.Module):
    def __init__(self, num_classes=1000):
        super(Resnet18, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.resnet.fc = torch.nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)