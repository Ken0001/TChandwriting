import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class DenseLayer(nn.Module):
    def __init__(self, input_features, growth_rate=32):
        super(DenseLayer, self).__init__()
        self.norm1 = nn.BatchNorm2d(input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(input_features, growth_rate*4, 1, stride=1, padding=0)

        self.norm2 = nn.BatchNorm2d(growth_rate*4)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(growth_rate*4, growth_rate, 3, stride=1, padding=1)

    def forward(self, x):
        out = self.conv1(self.relu1(self.norm1(x)))
        out = self.conv2(self.relu2(self.norm2(out)))
        out = torch.cat([x, out], 1)
        return out


class DenseBlock(nn.Module):
    def __init__(self, num_layers, input_features, growth_rate):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(DenseLayer(input_features+i*growth_rate, growth_rate))
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)

class TransitionLayer(nn.Module):
    def __init__(self, input_features, output_features):
        super(TransitionLayer, self).__init__()
        self.norm = nn.BatchNorm2d(input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(input_features, output_features, 1)

    def forward(self, x):
        out = self.conv(self.relu(self.norm(x)))
        return F.avg_pool2d(out, 2)


class DenseNet(nn.Module):
    def __init__(self, num_classes, growth_rate=32, block=(6, 12, 24, 16), compression_rate=0.5, first_features=64):
        super(DenseNet, self).__init__()
        # Top layers
        self.layers = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, 64, 5, stride=2, padding=1)),
            ('norm0', nn.BatchNorm2d(64)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(2, stride=2, padding=1)),
        ]))
        # Denseblocks
        features = first_features
        for i, num_layers in enumerate(block):
            dense_block = DenseBlock(num_layers, features, growth_rate)
            self.layers.add_module('Dense Block%d' %(i+1), dense_block)
            features = features+num_layers*growth_rate
            if i != 3:
                transition_layer = TransitionLayer(features, features//2)
                self.layers.add_module('Transition%d' %(i+1), transition_layer)
                features = features//2

        self.norm = nn.BatchNorm2d(features)
        self.relu = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(features, num_classes)

    def forward(self, x):
        out = self.layers(x)
        out = self.relu(self.norm(out))
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        return F.log_softmax(self.classifier(out), dim=1)

if __name__=='__main__':
    model = DenseNet(512, block=(6, 12, 48, 32))
    print(model)