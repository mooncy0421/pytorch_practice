from sklearn.utils import shuffle
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms     # For data augmentation

weight_decay = 0.0001
momentum = 0.9

batch_size = 128

# https://tutorials.pytorch.kr/beginner/blitz/cifar10_tutorial.html
transform = transforms.Compose([transforms.ToTensor(), 
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=False)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=False)z
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer'
           'dog', 'frog', 'horse', 'ship', 'truck')

class BuildingBlock(nn.Module):
    def __init__(self, input_channel, output_channel, stride=1):
        super(BuildingBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=output_channel, out_channels=output_channel, kernel_size=3, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(output_channel)
        self.conv3 = nn.Conv2d(in_channels=output_channel, out_channels=output_channel*4, kernel_size=1, stride=stride, bias=False)
        self.bn3 = nn.BatchNorm2d(output_channel*4)

    def forward(self, x):
        shortcut = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out)) + x
        out = self.relu(out)

        return out
        

class ResNet(nn.Module):
    def __init__(self, num_blocks, input_channel, num_classes=10):
        super(ResNet, self).__init__()
        self.input_channel = input_channel
        self.conv1 = nn.Conv2d(3, 64, 7, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.make_layer(64, num_blocks[0])
        self.layer2 = self.make_layer(128, num_blocks[1])
        self.layer3 = self.make_layer(256, num_blocks[2])
        self.layer4 = self.make_layer(512, num_blocks[3])
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*4, num_classes)

    def make_layer(self, num_channels, num_blocks):
        layers = []
        for _ in range(num_blocks):
            layers.append(BuildingBlock(self.input_channel, num_channels))
            self.input_channel = num_channels * 4
        return nn.Sequential(*layers)

    def forward(self, x):
        print(x.shape)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        print(out.shape)
        out = self.layer1(out)
        print(out.shape)
        out = self.layer2(out)
        print(out.shape)
        out = self.layer3(out)
        print(out.shape)
        out = self.layer4(out)
        print(out.shape)

        out = self.avgpool(out)
        print(out.shape)
        out = self.fc(out)
        print(out.shape)
        return out

resnet50 = ResNet([3,4,6,3], 64)
input = torch.randn(1,3,64,64)
out = resnet50(input)