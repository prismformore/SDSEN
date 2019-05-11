import torch
from torch import nn
from torch import autograd
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from groupy.gconv.pytorch_gconv.splitgconv2d import P4ConvZ2, P4ConvP4
import settings


class SEBlock(nn.Module):
    def __init__(self, input_dim, reduction):
        super().__init__()
        mid = int(input_dim / reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, reduction),
            nn.ReLU(inplace=True),
            nn.Linear(reduction, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class NoSEBlock(nn.Module):
    def __init__(self, input_dim, reduction):
        super().__init__()

    def forward(self, x):
        return x


SE = SEBlock if settings.use_se else NoSEBlock


class DSEN(nn.Module):
    def __init__(self):
        super().__init__()
        channel = settings.channel
        self.channel = channel
        self.pre_layer = P4CZ2(3, channel)
        self.re_layer = nn.ModuleList(  
            [P4ConvP4(channel, channel, kernel_size=5, padding=2) for i in range(settings.depth - 3)])
        self.dec = nn.Sequential(
            nn.Conv2d(channel*4, channel, 5, padding=2),
            SE(channel, channel//2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channel, 3, 1))
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        ori = x
        oups = []

        x = self.pre_layer(x)
        for ii, lay in enumerate(self.re_layer):
            idn = x
            x = lay(x)
            x = idn + x
            x = self.relu(x)

        x = x.view(x.shape[0], -1, x.shape[3], x.shape[4])
        x = self.dec(x)
        oups.append(x)
        x = ori - x

        return oups


class SDSEN(nn.Module):
    def __init__(self):
        super().__init__()
        channel = settings.channel
        self.channel = channel
        self.pre_layer = P4CZ2(3, channel)
        self.re_layer = nn.ModuleList(  
            [P4ConvP4(channel*2, channel, kernel_size=5, padding=2) for i in range(settings.depth - 3)])

        self.connect_layers = nn.ModuleList([P4CP4(channel, channel) for i in range(settings.depth - 3)])

        self.dec = nn.Sequential(
            nn.Conv2d(channel*4, channel, 5, padding=2),
            SE(channel, channel//2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channel, 3, 1))
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        ori = x
        oups = []
        old_status =  [torch.zeros((x.shape[0], self.channel, 4, x.shape[2], x.shape[3])).cuda() for i in range(settings.depth -3)]

        for _i in range(settings.stage_num):
            status = []

            x = self.pre_layer(x)
            for ii, lay in enumerate(self.re_layer):
                idn = x
                connect_conv = self.connect_layers[ii]
                connect_fm = connect_conv(old_status[ii])
                x = torch.cat([x, connect_fm], dim=1)
                x = lay(x)
                x = idn + x
                x = self.relu(x)
                status.append(x)

            x = x.view(x.shape[0], -1, x.shape[3], x.shape[4])
            x = self.dec(x)
            oups.append(x)
            x = ori - x

            old_status = status.copy()
        return oups



class P4CZ2(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv0 = P4ConvZ2(in_channel, out_channel, kernel_size=5, padding=2)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.conv0(x)

        return self.relu(x)

class P4CP4(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv0 = P4ConvP4(in_channel, out_channel, kernel_size=5, padding=2)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.conv0(x)

        return self.relu(x)


if __name__ == '__main__':
    ts = torch.Tensor(16, 3, 64, 64)
    vr = Variable(ts)
    net = SDSEN()
    print(net)
    oups = net(vr)
    for oup in oups:
        print(oup.size())

