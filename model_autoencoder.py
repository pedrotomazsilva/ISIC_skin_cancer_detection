import torch
import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, input_size, n_channels_in, n_channels_out, kernel_size, padding, stride=1, dilation=1):
        super(DepthwiseSeparableConv, self).__init__()
        if padding == 'same':
            padding = int(0.5*((input_size-1)*stride-input_size+kernel_size+(kernel_size-1)*(dilation-1)))
            self.depthwise = nn.Conv2d(n_channels_in, n_channels_in, kernel_size=kernel_size,
                                       padding=padding, groups=n_channels_in, stride=stride)
        else:
            self.depthwise = nn.Conv2d(n_channels_in, n_channels_in, kernel_size=kernel_size,
                                       padding=padding, groups=n_channels_in, stride=stride)
        self.pointwise = nn.Conv2d(n_channels_in, n_channels_out, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class TransposeConv(nn.Module):
    def __init__(self, input_size, n_channels_in, n_channels_out, kernel_size, stride, padding):
        super(TransposeConv, self).__init__()
        if padding == 'same':
            padding = 0.5*((input_size-1)*stride+kernel_size-input_size)
            self.transconv = nn.ConvTranspose2d(n_channels_in, n_channels_out, kernel_size=kernel_size,
                                                padding=padding, stride=stride)
        else:
            self.transconv = nn.ConvTranspose2d(n_channels_in, n_channels_out, kernel_size=kernel_size,
                                       padding=padding, stride=stride)

    def forward(self, x):
        return self.transconv(x)


class Autoencoder(nn.Module):
    def __init__(self, input_size=256):
        super(Autoencoder, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.spatial_dropout = nn.Dropout2d(p=0.5, inplace=False)
        self.relu = nn.ReLU(inplace=False)
        self.sigmoid = nn.Sigmoid()

        self.conv1 = DepthwiseSeparableConv(input_size, n_channels_in=3, n_channels_out=64,
                                            kernel_size=3, padding='same', stride=1)
        self.batch_norm_1 = nn.InstanceNorm2d(num_features=64)
        self.conv2 = DepthwiseSeparableConv(input_size, n_channels_in=64, n_channels_out=64,
                                              kernel_size=2, padding=0, stride=2)
        self.batch_norm_2 = nn.InstanceNorm2d(num_features=64)

        self.conv3 = DepthwiseSeparableConv(input_size / 2, n_channels_in=64, n_channels_out=64,
                                              kernel_size=2, padding=0, stride=2)
        self.batch_norm_3 = nn.InstanceNorm2d(num_features=64)
        self.conv4 = DepthwiseSeparableConv(input_size / 4, n_channels_in=64, n_channels_out=64,
                                            kernel_size=2, padding=0, stride=2)
        self.batch_norm_4 = nn.InstanceNorm2d(num_features=64)

        self.conv5 = DepthwiseSeparableConv(input_size / 8, n_channels_in=64, n_channels_out=64,
                                              kernel_size=2, padding=0, stride=2)
        self.batch_norm_5 = nn.InstanceNorm2d(num_features=256)
        self.conv6 = DepthwiseSeparableConv(input_size / 16, n_channels_in=64, n_channels_out=32,
                                              kernel_size=2, padding=0, stride=2)
        self.batch_norm_6 = nn.InstanceNorm2d(num_features=32)




        self.upconv7 = TransposeConv(input_size/32, n_channels_in=32, n_channels_out=64,
                                       kernel_size=2, stride=2, padding=0)
        self.batch_norm_7 = nn.InstanceNorm2d(num_features=64)
        self.upconv8 = TransposeConv(input_size / 16, n_channels_in=64, n_channels_out=64,
                                     kernel_size=2, stride=2, padding=0)
        self.batch_norm_8 = nn.InstanceNorm2d(num_features=64)
        self.upconv9 = TransposeConv(input_size / 8, n_channels_in=64, n_channels_out=64,
                                     kernel_size=2, stride=2, padding=0)
        self.batch_norm_9 = nn.InstanceNorm2d(num_features=64)

        self.upconv10 = TransposeConv(input_size / 4, n_channels_in=64, n_channels_out=64,
                                      kernel_size=2, stride=2, padding=0)
        self.batch_norm_10 = nn.InstanceNorm2d(num_features=64)

        self.upconv11 = TransposeConv(input_size / 2, n_channels_in=64, n_channels_out=64,
                                      kernel_size=2, stride=2, padding=0)
        self.batch_norm_11 = nn.InstanceNorm2d(num_features=64)

        self.upconv12 = DepthwiseSeparableConv(input_size, n_channels_in=64, n_channels_out=3,
                                               kernel_size=3, padding='same', stride=1)

    def forward(self, input_img):

        # Encoder
        img_1 = self.batch_norm_1(self.relu(self.conv1(input_img)))
        img_2 = self.batch_norm_2(self.relu(self.conv2(img_1)))
        img_3 = self.batch_norm_3(self.relu(self.conv3(img_2)))
        img_4 = self.batch_norm_4(self.relu(self.conv4(img_3)))
        img_5 = self.batch_norm_5(self.relu(self.conv5(img_4)))
        img_6 = self.batch_norm_6(self.relu(self.conv6(img_5)))

        img_7 = self.batch_norm_7(self.relu(self.upconv7(img_6)))
        img_8 = self.batch_norm_8(self.relu(self.upconv8(img_7)))
        img_9 = self.batch_norm_9(self.relu(self.upconv9(img_8)))
        img_10 = self.batch_norm_10(self.relu(self.upconv10(img_9)))
        img_11 = self.batch_norm_11(self.relu(self.upconv11(img_10)))
        img_12 = self.sigmoid(self.upconv12(img_11))

        return img_12


class Decoder(nn.Module):
    def __init__(self, input_size=256):
        super(Decoder, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.spatial_dropout = nn.Dropout2d(p=0.5, inplace=False)
        self.relu = nn.ReLU(inplace=False)
        self.sigmoid = nn.Sigmoid()

        self.upconv7 = TransposeConv(input_size/32, n_channels_in=32, n_channels_out=64,
                                       kernel_size=2, stride=2, padding=0)
        self.batch_norm_7 = nn.InstanceNorm2d(num_features=64)
        self.upconv8 = TransposeConv(input_size / 16, n_channels_in=64, n_channels_out=64,
                                     kernel_size=2, stride=2, padding=0)
        self.batch_norm_8 = nn.InstanceNorm2d(num_features=64)
        self.upconv9 = TransposeConv(input_size / 8, n_channels_in=64, n_channels_out=64,
                                     kernel_size=2, stride=2, padding=0)
        self.batch_norm_9 = nn.InstanceNorm2d(num_features=64)

        self.upconv10 = TransposeConv(input_size / 4, n_channels_in=64, n_channels_out=64,
                                      kernel_size=2, stride=2, padding=0)
        self.batch_norm_10 = nn.InstanceNorm2d(num_features=64)

        self.upconv11 = TransposeConv(input_size / 2, n_channels_in=64, n_channels_out=64,
                                      kernel_size=2, stride=2, padding=0)
        self.batch_norm_11 = nn.InstanceNorm2d(num_features=64)

        self.upconv12 = DepthwiseSeparableConv(input_size, n_channels_in=64, n_channels_out=3,
                                               kernel_size=3, padding='same', stride=1)

    def forward(self, input_img):

        img_7 = self.batch_norm_7(self.relu(self.upconv7(input_img)))
        img_8 = self.batch_norm_8(self.relu(self.upconv8(img_7)))
        img_9 = self.batch_norm_9(self.relu(self.upconv9(img_8)))
        img_10 = self.batch_norm_10(self.relu(self.upconv10(img_9)))
        img_11 = self.batch_norm_11(self.relu(self.upconv11(img_10)))
        img_12 = self.sigmoid(self.upconv12(img_11))

        return img_12


class Encoder(nn.Module):
    def __init__(self, input_size=256):
        super(Encoder, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.spatial_dropout = nn.Dropout2d(p=0.5, inplace=False)
        self.relu = nn.ReLU(inplace=False)
        self.sigmoid = nn.Sigmoid()

        self.conv1 = DepthwiseSeparableConv(input_size, n_channels_in=3, n_channels_out=64,
                                            kernel_size=3, padding='same', stride=1)
        self.batch_norm_1 = nn.InstanceNorm2d(num_features=64)
        self.conv2 = DepthwiseSeparableConv(input_size, n_channels_in=64, n_channels_out=64,
                                              kernel_size=2, padding=0, stride=2)
        self.batch_norm_2 = nn.InstanceNorm2d(num_features=64)

        self.conv3 = DepthwiseSeparableConv(input_size / 2, n_channels_in=64, n_channels_out=64,
                                              kernel_size=2, padding=0, stride=2)
        self.batch_norm_3 = nn.InstanceNorm2d(num_features=64)
        self.conv4 = DepthwiseSeparableConv(input_size / 4, n_channels_in=64, n_channels_out=64,
                                            kernel_size=2, padding=0, stride=2)
        self.batch_norm_4 = nn.InstanceNorm2d(num_features=64)

        self.conv5 = DepthwiseSeparableConv(input_size / 8, n_channels_in=64, n_channels_out=64,
                                              kernel_size=2, padding=0, stride=2)
        self.batch_norm_5 = nn.InstanceNorm2d(num_features=256)
        self.conv6 = DepthwiseSeparableConv(input_size / 16, n_channels_in=64, n_channels_out=32,
                                              kernel_size=2, padding=0, stride=2)
        self.batch_norm_6 = nn.InstanceNorm2d(num_features=32)
        self.linear = nn.Linear(2048, 1)

    def forward(self, input_img):

        # Encoder
        img_1 = self.batch_norm_1(self.relu(self.conv1(input_img)))
        img_2 = self.batch_norm_2(self.relu(self.conv2(img_1)))
        img_3 = self.batch_norm_3(self.relu(self.conv3(img_2)))
        img_4 = self.batch_norm_4(self.relu(self.conv4(img_3)))
        img_5 = self.batch_norm_5(self.relu(self.conv5(img_4)))
        img_6 = self.batch_norm_6(self.relu(self.conv6(img_5)))
        img_6_vec = img_6.view(-1, 2048)  # channels_out*img_size/32*img_size/32
        p = self.sigmoid(self.linear(img_6_vec))

        return p


def dice_coef_loss(input_batch, target_batch):

    total_loss = 0
    for i, img in enumerate(input_batch):
        smooth = 1.

        iflat = img.view(-1)
        tflat = target_batch[i].view(-1)
        intersection = (iflat * tflat).sum()

        dice = - ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))
        total_loss += dice
    loss = total_loss/len(input_batch)

    return loss


