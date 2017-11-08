from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1,
                 activation=nn.ReLU(True), norm=nn.InstanceNorm2d, pad_type=nn.ReflectionPad2d):
        super(ConvBlock, self).__init__()
        self.pad = pad_type(padding)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride)
        self.norm = norm(out_channels)
        self.activation = activation

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, scale_factor=2,
                 activation=nn.ReLU(True), norm=nn.InstanceNorm2d, pad_type=nn.ReflectionPad2d):
        super(UpBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor)
        self.pad = pad_type(padding)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.norm = norm(out_channels)
        self.activation = activation

    def forward(self, x):
        x = self.upsample(x)
        x = self.pad(x)
        x = self.conv(x)
        x = self.norm(x)
        return self.activation(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 activation=nn.ReLU(True), norm=nn.InstanceNorm2d,
                 pad_type=nn.ReflectionPad2d):
        super(ResBlock, self).__init__()
        self.pad = pad_type(1)
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3)
        self.norm = norm(out_channels)
        self.dropout = nn.Dropout2d(0.5)
        self.activation = activation

    def forward(self, x):
        fx = self.pad(x)
        fx = self.conv3x3(fx)
        fx = self.norm(fx)
        fx = self.activation(fx)
        fx = self.dropout(fx)
        fx = self.pad(fx)
        fx = self.conv3x3(fx)
        fx = self.norm(fx)
        return fx + x
