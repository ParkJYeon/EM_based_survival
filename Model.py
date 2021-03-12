import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvolutionNN(nn.Module):
    def __init__(self):
        super(ConvolutionNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=80, kernel_size=1, stride=2)
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        self.pool1 = nn.MaxPool2d(6, stride=4)
        self.conv2 = nn.Conv2d(in_channels=80, out_channels=120, kernel_size=5, stride=1)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        self.pool2 = nn.MaxPool2d(3, stride=2)
        self.conv3 = nn.Conv2d(in_channels=120, out_channels=160, kernel_size=3, stride=1)
        torch.nn.init.xavier_uniform_(self.conv3.weight)
        self.conv4 = nn.Conv2d(in_channels=160, out_channels=200, kernel_size=3, stride=1)
        torch.nn.init.xavier_uniform_(self.conv4.weight)
        self.pool3 = nn.MaxPool2d(3, stride=2)

        self.conv_module = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            #nn.LocalResponseNorm(80),
            nn.BatchNorm2d(80),
            self.pool1,
            self.conv2,
            nn.ReLU(),
            #nn.LocalResponseNorm(120),
            nn.BatchNorm2d(120),
            self.pool2,
            self.conv3,
            nn.ReLU(),
            self.conv4,
            nn.ReLU(),
            self.pool3
        )

        self.fc1 = nn.Linear(200 * 4 * 4, 320)
        self.fc2 = nn.Linear(320, 320)
        self.fc3 = nn.Linear(320, 2)

        self.fc_module = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            nn.Dropout(p=0.5),
            self.fc2,
            nn.ReLU(),
            nn.Dropout(p=0.5),
            self.fc3
        )

        self.params = list(self.parameters())

        if torch.cuda.is_available():
            self.conv_module = self.conv_module.cuda()
            self.fc_module = self.fc_module.cuda()

    def forward(self, x):
        out = self.conv_module(x)
        out = out.view(-1, 200 * 4 * 4)
        out = self.fc_module(out)
        out = F.softmax(out, dim=1)
        return out
