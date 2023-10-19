"""
derived from https://github.com/milesial/Pytorch-UNet/blob/master/unet/
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class mini_U_Net(nn.Module):
    def __init__(self,img_ch=1,output_ch=1):
        super().__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)

        self.Up2 = up_conv(ch_in=128)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        # decoding + concat path
        d2 = self.Up2(x2)
        d2 = up_cat(x1,d2)
        d2 = self.Up_conv2(d2)

        return self.Conv_1x1(d2)

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
            )

    def forward(self,x):
        return self.double_conv(x)

class up_conv(nn.Module):
    def __init__(self,ch_in):
        super().__init__()
        self.up = nn.ConvTranspose2d(ch_in, ch_in // 2, kernel_size=2, stride=2)

    def forward(self,x):
        return self.up(x)

def up_cat(x1,x2):
    return torch.cat((x1,up_pad(x1,x2)),dim=1)

def up_pad(x1,x2):
    diffY = x1.size()[2] - x2.size()[2]
    diffX = x1.size()[3] - x2.size()[3]
    return F.pad(x2, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

# EOF
