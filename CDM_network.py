"""
derived from U-net:
https://github.com/milesial/Pytorch-UNet/blob/master/unet/
mini version for dev
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# mini U-NET with only single bottom layer (for dev)
class mini_U_Net(nn.Module):
    def __init__(self,img_ch=1,output_ch=1,bilinear=True):
        super().__init__()
        fac = 2 if bilinear else 1

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128 // fac)

        self.Up2 = up_conv(ch_in=128, bilinear=bilinear)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64, bilinear=bilinear)

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
    """ added bilinear threatment """
    def __init__(self,ch_in,ch_out,bilinear=False):
        super().__init__()
        ch_mid = ch_out if not bilinear else ch_in // 2
        self.double_conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_mid, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_mid, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
            )

    def forward(self,x):
        return self.double_conv(x)

class up_conv(nn.Module):
    """ added bilinear threatment """
    def __init__(self,ch_in,bilinear=False):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(ch_in, ch_in // 2, kernel_size=2, stride=2)

    def forward(self,x):
        return self.up(x)

def up_cat(x1,x2):
    """
    fixes dimension problems with concatanate 
    if you have padding issues, see
    https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
    https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
    """
    return torch.cat((x1,up_pad(x1,x2)),dim=1)

def up_pad(x1,x2):
    """ fixes dimension problems with padding 
    pads smaller x2 to larger x1 so x1 and x2 have the same dimensions
    """
    diffY = x1.size()[2] - x2.size()[2]
    diffX = x1.size()[3] - x2.size()[3]
    return F.pad(x2, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

# EO
