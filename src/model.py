import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


class Yolo(nn.Module):
    def __init__(self, grid_num, bbox_num, class_num=20, is_train=True):
        super(Yolo, self).__init__()

        # params
        self.grid_num = grid_num
        self.bbox_num = bbox_num
        self.class_num = class_num
        self.is_train = is_train

        # layer1
        self.conv1_1 = ConvWithBN(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # layer2
        self.conv2_1 = ConvWithBN(in_channels=64, out_channels=192, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # layer3
        self.conv3_1 = ConvWithBN(in_channels=192, out_channels=128, kernel_size=1)
        self.conv3_2 = ConvWithBN(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv3_3 = ConvWithBN(in_channels=256, out_channels=256, kernel_size=1)
        self.conv3_4 = ConvWithBN(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # layer4
        self.conv4_1 = ConvWithBN(in_channels=512, out_channels=256, kernel_size=1)
        self.conv4_2 = ConvWithBN(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv4_3 = ConvWithBN(in_channels=512, out_channels=256, kernel_size=1)
        self.conv4_4 = ConvWithBN(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv4_5 = ConvWithBN(in_channels=512, out_channels=256, kernel_size=1)
        self.conv4_6 = ConvWithBN(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv4_7 = ConvWithBN(in_channels=512, out_channels=256, kernel_size=1)
        self.conv4_8 = ConvWithBN(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv4_9 = ConvWithBN(in_channels=512, out_channels=512, kernel_size=1)
        self.conv4_10 = ConvWithBN(in_channels=512, out_channels=1024, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # layer5
        self.conv5_1 = ConvWithBN(in_channels=1024, out_channels=512, kernel_size=1)
        self.conv5_2 = ConvWithBN(in_channels=512, out_channels=1024, kernel_size=3, padding=1)
        self.conv5_3 = ConvWithBN(in_channels=1024, out_channels=512, kernel_size=1)
        self.conv5_4 = ConvWithBN(in_channels=512, out_channels=1024, kernel_size=3, padding=1)
        self.conv5_5 = ConvWithBN(in_channels=1024, out_channels=1024, kernel_size=3, padding=1)
        self.conv5_6 = ConvWithBN(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=1)

        # layer6
        self.conv6_1 = ConvWithBN(in_channels=1024, out_channels=1024, kernel_size=3, padding=1)
        self.conv6_2 = ConvWithBN(in_channels=1024, out_channels=1024, kernel_size=3, padding=1)

        # layer7
        self.fc7 = nn.Linear(self.grid_num * self.grid_num * 1024, 4096)
        self.dropout7 = nn.Dropout(p=0.3)

        # layer8
        self.fc8 = nn.Linear(4096, self.grid_num * self.grid_num * (5 * self.bbox_num + self.class_num))

    def forward(self, x):
        # layer1
        x = self.conv1_1(x)
        x = self.pool1(x)

        # layer2
        x = self.conv2_1(x)
        x = self.pool2(x)

        # layer3
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.conv3_4(x)
        x = self.pool3(x)

        # layer4
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.conv4_4(x)
        x = self.conv4_5(x)
        x = self.conv4_6(x)
        x = self.conv4_7(x)
        x = self.conv4_8(x)
        x = self.conv4_9(x)
        x = self.conv4_10(x)
        x = self.pool4(x)

        # layer5
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.conv5_4(x)
        x = self.conv5_5(x)
        x = self.conv5_6(x)

        # layer6
        x = self.conv6_1(x)
        x = self.conv6_2(x)

        # layer7
        x = x.view(-1, self.num_flat_features(x))
        x = F.leaky_relu(self.fc7(x), negative_slope=0.1)
        if self.is_train:
            x = self.dropout7(x)

        # layer8
        x = self.fc8(x)
        x = x.view(-1, 5 * self.bbox_num + self.class_num, self.grid_num, self.grid_num)
        x1 = torch.sigmoid(x[:, :5 * self.bbox_num])
        x2 = F.softmax(x[:, 5 * self.bbox_num:], dim=1)
        x = torch.cat([x1, x2], dim=1)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class ConvWithBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvWithBN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.bn(self.conv(x))
        x = F.leaky_relu(x, negative_slope=0.1)
        return x
