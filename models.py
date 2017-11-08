from torch import nn
from layers import ConvBlock, ResBlock, UpBlock


class Generator128(nn.Module):
    """
    1x large kernel conv
    2x stride two convs
    6x resnet processing blocks
    2x upsampling blocks
    1x large kernal conv out + tanh
    """
    def __init__(self):
        super(Generator128, self).__init__()
        self.conv_in = ConvBlock(3, 64, kernel_size=7, padding=3)

        self.encode1 = ConvBlock(64, 128, kernel_size=3, stride=2, padding=1)
        self.encode2 = ConvBlock(128, 256, kernel_size=3, stride=2, padding=1)

        self.res1 = ResBlock(256, 256)
        self.res2 = ResBlock(256, 256)
        self.res3 = ResBlock(256, 256)
        self.res4 = ResBlock(256, 256)
        self.res5 = ResBlock(256, 256)
        self.res6 = ResBlock(256, 256)

        self.decode1 = UpBlock(256, 128, kernel_size=3, padding=1)
        self.decode2 = UpBlock(128, 64, kernel_size=3, padding=1)

        self.conv_out = ConvBlock(64, 3, kernel_size=7, padding=3, activation=nn.Tanh())

    def forward(self, x):
        x = self.conv_in(x)  # (64, 128, 128)

        x = self.encode1(x)  # (128, 64, 64)
        x = self.encode2(x)  # (256, 32, 32)

        x = self.res1(x)  # (256, 32, 32) # res blocks do not change size
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.res6(x)

        x = self.decode1(x)  # (128, 64, 64)
        x = self.decode2(x)  # (64, 128, 128)

        return self.conv_out(x)  # (3, 128, 128)


class PatchGan(nn.Module):
    """
    Fully convolutional patchgan discriminator
    """

    def __init__(self, activation=nn.LeakyReLU(0.2, True)):
        super(PatchGan, self).__init__()
        self.activation = activation
        self.conv_in = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)

        self.layer_1 = ConvBlock(64, 128, kernel_size=4, stride=2,
                                 padding=1, activation=activation)
        self.layer_2 = ConvBlock(128, 256, kernel_size=4, stride=2,
                                 padding=1, activation=activation)
        self.layer_3 = ConvBlock(256, 512, kernel_size=4, stride=2,
                                 padding=1, activation=activation)

        self.conv_out = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)

    def forward(self, x):
        # x = (3, 128, 128)
        x = self.activation(self.conv_in(x))  # (64, 64, 64)
        x = self.layer_1(x)  # (128, 32, 32)
        x = self.layer_2(x)  # (256, 16, 16)
        x = self.layer_3(x)  # (512, 8, 8)
        return self.conv_out(x)  # (1, 7, 7)
