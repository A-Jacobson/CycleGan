from torch import nn
from layers import ConvBlock, ResBlock, UpBlock


class ResnetGenerator(nn.Module):
    """
    1x large kernel conv
    2x stride two convs
    6-9x resnet processing blocks
    2x upsampling blocks
    1x large kernal conv out + tanh
    """
    def __init__(self, num_resblocks=6):
        super(ResnetGenerator, self).__init__()
        self.conv_in = ConvBlock(3, 32, kernel_size=7, padding=3)

        self.encode1 = ConvBlock(32, 64, kernel_size=3, stride=2, padding=1)
        self.encode2 = ConvBlock(64, 128, kernel_size=3, stride=2, padding=1)

        self.resblocks = nn.Sequential(*(ResBlock(128, 128) for _ in range(num_resblocks)))

        self.decode1 = UpBlock(128, 64, kernel_size=3, padding=1)
        self.decode2 = UpBlock(64, 32, kernel_size=3, padding=1)

        self.conv_out = ConvBlock(32, 3, kernel_size=7, padding=3, activation=nn.Tanh())

    def forward(self, x):
        x = self.conv_in(x)  # (32, 128, 128) or (32, 256, 256)

        x = self.encode1(x)  # (64, 64, 64) | (64, 128, 128)
        x = self.encode2(x)  # (128, 32, 32) | (128, 64, 64)
        # res blocks do not change size (6-9 blocks)
        x = self.resblocks(x)  # (128, 32, 32) | (128, 64, 64)

        x = self.decode1(x)  # (64, 64, 64) | (64, 128, 128)
        x = self.decode2(x)  # (32, 128, 128) | (32, 256, 256)

        return self.conv_out(x)  # (3, 128, 128) | (3, 256, 256)


class PatchDiscriminator(nn.Module):
    """
    Fully convolutional patchgan discriminator
    """

    def __init__(self, activation=nn.LeakyReLU(0.2, True)):
        super(PatchDiscriminator, self).__init__()
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
        x = self.activation(self.conv_in(x))  # (64, 64, 64) | (64, 128, 128)
        x = self.layer_1(x)  # (128, 32, 32) | (128, 64, 64)
        x = self.layer_2(x)  # (256, 16, 16) | (256, 32, 32)
        x = self.layer_3(x)  # (512, 8, 8)  | (512, 16, 16)
        return self.conv_out(x)  # (1, 7, 7) |
