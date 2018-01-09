import torch
import torch.nn as nn


class HidingNet(nn.Module):
    def __init__(self, ngpu, nc=6, nhf=50):
        super(HidingNet, self).__init__()
        self.ngpu = ngpu
        # input is (6) x 128 x 128
        self.conv1_3_3 = nn.Conv2d(nc, nhf, 3, 1, 1, bias=False)
        self.conv1_5_5 = nn.Conv2d(nc, nhf, 5, 1, 2, bias=False)
        self.BN1 = nn.BatchNorm2d(nhf * 2)
        self.RELU1 = nn.ReLU(True)

        # input is 100*128*128
        self.conv2_3_3 = nn.Conv2d(2 * nhf, nhf, 3, 1, 1, bias=False)
        self.conv2_5_5 = nn.Conv2d(2 * nhf, nhf, 5, 1, 2, bias=False)
        self.BN2 = nn.BatchNorm2d(nhf * 2)
        self.RELU2 = nn.ReLU(True)

        # input is 100*128*128
        self.conv3_3_3 = nn.Conv2d(2 * nhf, nhf, 3, 1, 1, bias=False)
        self.conv3_5_5 = nn.Conv2d(2 * nhf, nhf, 5, 1, 2, bias=False)
        self.BN3 = nn.BatchNorm2d(nhf * 2)
        self.RELU3 = nn.ReLU(True)

        # input is 100*128*128
        self.conv4_3_3 = nn.Conv2d(2 * nhf, nhf, 3, 1, 1, bias=False)
        self.conv4_5_5 = nn.Conv2d(2 * nhf, nhf, 5, 1, 2, bias=False)
        self.BN4 = nn.BatchNorm2d(nhf * 2)
        self.RELU4 = nn.ReLU(True)

        # input is 100*128*128
        self.conv5_3_3 = nn.Conv2d(2 * nhf, nhf, 3, 1, 1, bias=False)
        self.conv5_5_5 = nn.Conv2d(2 * nhf, nhf, 5, 1, 2, bias=False)
        self.BN5 = nn.BatchNorm2d(nhf * 2)
        self.RELU5 = nn.ReLU(True)

        # input is 100*128*128
        self.conv6_1_1 = nn.Conv2d(2 * nhf, 3, 1, 1, 0, bias=False)
        self.Tanh6 = nn.Tanh()

    def forward(self, input):
        l1_3 = self.conv1_3_3(input)
        l1_5 = self.conv1_5_5(input)
        l1_cat = torch.cat([l1_3, l1_5], 1)
        l1_bn = self.BN1(l1_cat)
        l1_out = self.RELU1(l1_bn)

        l2_3 = self.conv2_3_3(l1_out)
        l2_5 = self.conv2_5_5(l1_out)
        l2_cat = torch.cat([l2_3, l2_5], 1)
        l2_bn = self.BN2(l2_cat)
        l2_out = self.RELU2(l2_bn)

        l3_3 = self.conv3_3_3(l2_out)
        l3_5 = self.conv3_5_5(l2_out)
        l3_cat = torch.cat([l3_3, l3_5], 1)
        l3_bn = self.BN3(l3_cat)
        l3_out = self.RELU3(l3_bn)

        l4_3 = self.conv4_3_3(l3_out)
        l4_5 = self.conv4_5_5(l3_out)
        l4_cat = torch.cat([l4_3, l4_5], 1)
        l4_bn = self.BN4(l4_cat)
        l4_out = self.RELU4(l4_bn)

        l5_3 = self.conv5_3_3(l4_out)
        l5_5 = self.conv5_5_5(l4_out)
        l5_cat = torch.cat([l5_3, l5_5], 1)
        l5_bn = self.BN5(l5_cat)
        l5_out = self.RELU5(l5_bn)

        l6 = self.conv6_1_1(l5_out)
        output = self.Tanh6(l6)

        return output
