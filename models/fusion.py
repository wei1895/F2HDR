"""Stage-1 fusion network: blends the HDR candidates into a coarse HDR."""
import torch
import torch.nn as nn

from .network_utils import MergeHDRModule


class Fusion_Net(nn.Module):
    def __init__(self, c_in=30, c_out=5, c_mid=128):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(c_in, c_mid // 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c_mid // 4), nn.LeakyReLU(0.1, inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(c_mid // 4, c_mid // 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c_mid // 4), nn.LeakyReLU(0.1, inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(c_mid // 4, c_mid // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c_mid // 2), nn.LeakyReLU(0.1, inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(c_mid // 2, c_mid // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c_mid // 2), nn.LeakyReLU(0.1, inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(c_mid // 2, c_mid, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c_mid), nn.LeakyReLU(0.1, inplace=True))
        self.conv6 = nn.Sequential(
            nn.Conv2d(c_mid, c_mid, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c_mid), nn.LeakyReLU(0.1, inplace=True))
        self.conv7 = nn.Sequential(
            nn.ConvTranspose2d(c_mid, c_mid // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c_mid // 2), nn.LeakyReLU(0.1, inplace=True))
        self.conv8 = nn.Sequential(
            nn.Conv2d(c_mid, c_mid // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c_mid // 2), nn.LeakyReLU(0.1, inplace=True))
        self.conv9 = nn.Sequential(
            nn.ConvTranspose2d(c_mid // 2, c_mid // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c_mid // 2), nn.LeakyReLU(0.1, inplace=True))
        self.conv10 = nn.Sequential(
            nn.Conv2d(c_mid // 2 + c_mid // 4, c_mid // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c_mid // 2), nn.LeakyReLU(0.1, inplace=True))
        self.conv11 = nn.Sequential(
            nn.ConvTranspose2d(c_mid // 2, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.LeakyReLU(0.1, inplace=True))
        self.conv12 = nn.Sequential(nn.Conv2d(64, c_out, kernel_size=3, stride=1, padding=1))

        self.merge_HDR = MergeHDRModule()

    def forward(self, inputs, hdrs):
        # inputs: (B, 30, H, W) HDR+LDR sources; hdrs: 5 HDR candidates to blend.
        ### 1/2
        conv1 = self.conv1(inputs)
        conv2 = self.conv2(conv1)
        ### 1/4
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        ### 1/8
        conv5 = self.conv5(conv4)
        conv6 = self.conv6(conv5)
        ### 1/4
        conv7 = self.conv7(conv6)
        conv8 = self.conv8(torch.cat((conv7, conv4), dim=1))
        ### 1/2
        conv9 = self.conv9(conv8)
        conv10 = self.conv10(torch.cat((conv9, conv2), dim=1))
        ### output
        conv11 = self.conv11(conv10)
        conv12 = self.conv12(conv11)
        weights = torch.sigmoid(conv12)

        ws = torch.split(weights, 1, 1)
        hdr, _ = self.merge_HDR(ws, hdrs)
        return hdr
