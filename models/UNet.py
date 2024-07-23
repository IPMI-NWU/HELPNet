import copy
from typing import Optional, List
from torch.autograd import Variable
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from einops import rearrange, repeat


class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch, padding=0):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=padding, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_ch),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=padding, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.01),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, in_ch, out_ch, padding=1):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=padding, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.01),
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Unet_keep_size(nn.Module):

    def __init__(self, in_ch=1, out_ch=4):
        super(Unet_keep_size, self).__init__()

        filters = [64, 128, 256, 512, 1024]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0], padding=1)
        self.Conv2 = conv_block(filters[0], filters[1], padding=1)
        self.Conv3 = conv_block(filters[1], filters[2], padding=1)
        self.Conv4 = conv_block(filters[2], filters[3], padding=1)
        self.Conv5 = conv_block(filters[3], filters[4], padding=1)

        self.Up4 = up_conv(filters[4], 4, padding=1)
        self.Up_conv4 = conv_block(516, filters[3], padding=1)

        self.Up3 = up_conv(filters[3], 4, padding=1)
        self.Up_conv3 = conv_block(260, filters[2], padding=1)

        self.Up2 = up_conv(filters[2], filters[1], padding=1)
        self.Up_conv2 = conv_block(filters[2], filters[1], padding=1)

        self.Up1 = up_conv(filters[1], filters[0], padding=1)
        self.Up_conv1 = conv_block(filters[1], filters[0], padding=1)

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)
        self.Norm = nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.01)
        self.active = torch.nn.Softmax(dim=1)

    def forward(self, tensor_list):
        x1 = tensor_list

        e1 = self.Conv1(x1)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d4 = self.Up4(e5)
        d4 = torch.cat((d4, e4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Up1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.Up_conv1(d1)

        d0 = self.Conv(d1)
        norm_out = self.Norm(d0)
        out = self.active(norm_out)
        return out


class Unet_new(nn.Module):

    def __init__(self, in_ch=1, out_ch=4):
        super(Unet_new, self).__init__()

        filters = [32, 64, 128, 256, 512]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0], padding=1)
        self.Conv2 = conv_block(filters[0], filters[1], padding=1)
        self.Conv3 = conv_block(filters[1], filters[2], padding=1)
        self.Conv4 = conv_block(filters[2], filters[3], padding=1)
        self.Conv5 = conv_block(filters[3], filters[4], padding=1)

        self.Up4 = up_conv(filters[4], filters[3], padding=1)
        self.Up_conv4 = conv_block(filters[4], filters[3], padding=1)

        self.Up3 = up_conv(filters[3], filters[2], padding=1)
        self.Up_conv3 = conv_block(filters[3], filters[2], padding=1)

        self.Up2 = up_conv(filters[2], filters[1], padding=1)
        self.Up_conv2 = conv_block(filters[2], filters[1], padding=1)

        self.Up1 = up_conv(filters[1], filters[0], padding=1)
        self.Up_conv1 = conv_block(filters[1], filters[0], padding=1)

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)
        self.Norm = nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.01)
        self.active = torch.nn.Softmax(dim=1)

    def forward(self, tensor_list):
        x1 = tensor_list

        e1 = self.Conv1(x1)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d4 = self.Up4(e5)
        d4 = torch.cat((d4, e4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Up1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.Up_conv1(d1)

        d0 = self.Conv(d1)
        norm_out = self.Norm(d0)
        out = self.active(norm_out)
        return out


def build_UNet():
    return Unet_keep_size(in_ch=1, out_ch=4)
