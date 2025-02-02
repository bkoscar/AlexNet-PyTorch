import torch.nn as nn 
import torch.nn.functional as F
import torch

class AlexNet(nn.Module):

    def __init__(self, config):
        super(AlexNet, self).__init__()
        self.config = config
        self.features = self._make_features()
        self.classifier = self._make_classifier()

    def _make_features(self):
        config = self.config['features']
        layers = [
            nn.Conv2d(in_channels=config['in_channels1'], out_channels=config['out_channels1'], kernel_size=config['kernel_size1'], stride=config['stride1'], padding=config['padding1']),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=config['kernel_size_pool1'], stride=config['stride_pool1']),
            nn.Conv2d(in_channels=config['in_channels2'], out_channels=config['out_channels2'], kernel_size=config['kernel_size2'], stride=config['stride2'], padding=config['padding2']),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=config['kernel_size_pool2'], stride=config['stride_pool2']),
            nn.Conv2d(in_channels=config['in_channels3'], out_channels=config['out_channels3'], kernel_size=config['kernel_size3'], stride=config['stride3'], padding=config['padding3']),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=config['in_channels4'], out_channels=config['out_channels4'], kernel_size=config['kernel_size4'], stride=config['stride4'], padding=config['padding4']),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=config['in_channels5'], out_channels=config['out_channels5'], kernel_size=config['kernel_size5'], stride=config['stride5'], padding=config['padding5']),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=config['kernel_size_pool3'], stride=config['stride_pool3']),
        ]
        return nn.Sequential(*layers)

    def _make_classifier(self):
        config = self.config['classifier']
        layers = [
            nn.Dropout(p=0.5),
            nn.Linear(config['linear_input'], config['linear_hidden1']),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(config['linear_hidden1'], config['linear_hidden2']),
            nn.ReLU(inplace=True),
            nn.Linear(config['linear_hidden2'], config['num_classes']),
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x