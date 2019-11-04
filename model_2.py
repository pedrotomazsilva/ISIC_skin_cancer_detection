import torch
import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, input_size, n_channels_in, n_channels_out, kernel_size, padding, stride, dilation=1):
        super(DepthwiseSeparableConv, self).__init__()
        if padding == 'same':
            padding = int(0.5*((input_size-1)*stride-input_size+kernel_size+(kernel_size-1)*(dilation-1)))
            self.depthwise = nn.Conv2d(n_channels_in, n_channels_in, kernel_size=kernel_size,
                                       padding=padding, groups=n_channels_in)
        else:
            self.depthwise = nn.Conv2d(n_channels_in, n_channels_in, kernel_size=kernel_size,
                                       padding=padding, groups=n_channels_in)
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


class Unet(nn.Module):
    def __init__(self, input_size=256):
        super(Unet, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.spatial_dropout = nn.Dropout2d(p=0.5, inplace=False)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.sigmoid = nn.Sigmoid()

        self.conv1_1 = DepthwiseSeparableConv(input_size, n_channels_in=3, n_channels_out=64,
                                              kernel_size=3, padding='same', stride=1)
        self.batch_norm_1_1 = nn.InstanceNorm2d(num_features=64)
        self.conv1_2 = DepthwiseSeparableConv(input_size, n_channels_in=64, n_channels_out=64,
                                              kernel_size=3, padding='same', stride=1)
        self.convmax1_2 = DepthwiseSeparableConv(input_size, n_channels_in=64, n_channels_out=64,
                                                 kernel_size=2, padding=0, stride=2)
        self.batch_norm_1_2 = nn.InstanceNorm2d(num_features=64)

        self.conv2_1 = DepthwiseSeparableConv(input_size / 2, n_channels_in=64, n_channels_out=128,
                                              kernel_size=3, padding='same', stride=1)
        self.batch_norm_2_1 = nn.InstanceNorm2d(num_features=128)
        self.conv2_2 = DepthwiseSeparableConv(input_size / 2, n_channels_in=128, n_channels_out=128,
                                              kernel_size=3, padding='same', stride=1)
        self.convmax2_2 = DepthwiseSeparableConv(input_size / 2, n_channels_in=128, n_channels_out=128,
                                                 kernel_size=2, padding=0, stride=2)
        self.batch_norm_2_2 = nn.InstanceNorm2d(num_features=128)

        self.conv3_1 = DepthwiseSeparableConv(input_size / 4, n_channels_in=128, n_channels_out=256,
                                              kernel_size=3, padding='same', stride=1)
        self.batch_norm_3_1 = nn.InstanceNorm2d(num_features=256)
        self.conv3_2 = DepthwiseSeparableConv(input_size / 4, n_channels_in=256, n_channels_out=256,
                                              kernel_size=3, padding='same', stride=1)
        self.convmax3_2 = DepthwiseSeparableConv(input_size / 4, n_channels_in=256, n_channels_out=256,
                                                 kernel_size=2, padding=0, stride=2)
        self.batch_norm_3_2 = nn.InstanceNorm2d(num_features=256)

        self.conv4_1 = DepthwiseSeparableConv(input_size / 8, n_channels_in=256, n_channels_out=512,
                                              kernel_size=3, padding='same', stride=1)
        self.batch_norm_4_1 = nn.InstanceNorm2d(num_features=512)
        self.conv4_2 = DepthwiseSeparableConv(input_size / 8, n_channels_in=512, n_channels_out=512,
                                              kernel_size=3, padding='same', stride=1)
        self.convmax4_2 = DepthwiseSeparableConv(input_size / 8, n_channels_in=512, n_channels_out=512,
                                                 kernel_size=2, padding=0, stride=2)
        self.batch_norm_4_2 = nn.InstanceNorm2d(num_features=512)

        self.conv5_1 = DepthwiseSeparableConv(input_size / 16, n_channels_in=512, n_channels_out=1024,
                                              kernel_size=3, padding='same', stride=1)
        self.batch_norm_5_1 = nn.InstanceNorm2d(num_features=1024)
        self.conv5_2 = DepthwiseSeparableConv(input_size / 16, n_channels_in=1024, n_channels_out=1024,
                                              kernel_size=3, padding='same', stride=1)
        self.batch_norm_5_2 = nn.InstanceNorm2d(num_features=1024)

        self.upconv6_1 = TransposeConv(input_size/16, n_channels_in=1024, n_channels_out=512,
                                       kernel_size=2, stride=2, padding=0)
        self.batch_norm_up6_1 = nn.InstanceNorm2d(num_features=512)
        self.conv6_1 = DepthwiseSeparableConv(input_size / 8, n_channels_in=512+512, n_channels_out=512,
                                              kernel_size=3, padding='same', stride=1)
        self.batch_norm_conv6_1 = nn.InstanceNorm2d(num_features=512)
        self.conv6_2 = DepthwiseSeparableConv(input_size / 8, n_channels_in=512, n_channels_out=512,
                                              kernel_size=3, padding='same', stride=1)
        self.batch_norm_6_2 = nn.InstanceNorm2d(num_features=512)

        self.upconv7_1 = TransposeConv(input_size / 8, n_channels_in=512, n_channels_out=256,
                                       kernel_size=2, stride=2,
                                       padding=0)
        self.batch_norm_up7_1 = nn.InstanceNorm2d(num_features=256)
        self.conv7_1 = DepthwiseSeparableConv(input_size / 4, n_channels_in=256 + 256, n_channels_out=256,
                                              kernel_size=3, padding='same', stride=1)
        self.batch_norm_conv7_1 = nn.InstanceNorm2d(num_features=256)
        self.conv7_2 = DepthwiseSeparableConv(input_size / 4, n_channels_in=256, n_channels_out=256,
                                              kernel_size=3, padding='same', stride=1)
        self.batch_norm_7_2 = nn.InstanceNorm2d(num_features=256)

        self.upconv8_1 = TransposeConv(input_size / 4, n_channels_in=256, n_channels_out=128, kernel_size=2, stride=2,
                                       padding=0)
        self.batch_norm_up8_1 = nn.InstanceNorm2d(num_features=128)
        self.conv8_1 = DepthwiseSeparableConv(input_size / 2, n_channels_in=128 + 128, n_channels_out=128,
                                              kernel_size=3, padding='same', stride=1)
        self.batch_norm_conv8_1 = nn.InstanceNorm2d(num_features=128)
        self.conv8_2 = DepthwiseSeparableConv(input_size / 2, n_channels_in=128, n_channels_out=128,
                                              kernel_size=3, padding='same', stride=1)
        self.batch_norm_8_2 = nn.InstanceNorm2d(num_features=128)

        self.upconv9_1 = TransposeConv(input_size / 2, n_channels_in=128, n_channels_out=64, kernel_size=2, stride=2,
                                       padding=0)
        self.batch_norm_up9_1 = nn.InstanceNorm2d(num_features=64)
        self.conv9_1 = DepthwiseSeparableConv(input_size, n_channels_in=64 + 64, n_channels_out=64,
                                              kernel_size=3, padding='same', stride=1)
        self.batch_norm_conv9_1 = nn.InstanceNorm2d(num_features=64)
        self.conv9_2 = DepthwiseSeparableConv(input_size, n_channels_in=64, n_channels_out=64,
                                              kernel_size=3, padding='same', stride=1)
        self.batch_norm_9_2 = nn.InstanceNorm2d(num_features=64)

        self.conv9_3 = DepthwiseSeparableConv(input_size, n_channels_in=64, n_channels_out=2,
                                              kernel_size=3, padding='same', stride=1)
        self.batch_norm_9_3 = nn.InstanceNorm2d(num_features=2)

        self.conv9_4 = DepthwiseSeparableConv(input_size, n_channels_in=2, n_channels_out=1,
                                              kernel_size=1, padding='same', stride=1)

    def forward(self, input_img):

        # Encoder
        img_1_1 = self.batch_norm_1_1(self.relu(self.conv1_1(input_img)))
        img_1_2 = self.batch_norm_1_2(self.relu(self.conv1_2(img_1_1)))
        img_1_3 = self.convmax1_2(img_1_2)

        img_2_1 = self.batch_norm_2_1(self.relu(self.conv2_1(img_1_3)))
        img_2_2 = self.batch_norm_2_2(self.relu(self.conv2_2(img_2_1)))
        img_2_3 = self.convmax2_2(img_2_2)

        img_3_1 = self.batch_norm_3_1(self.relu(self.conv3_1(img_2_3)))
        img_3_2 = self.batch_norm_3_2(self.relu(self.conv3_2(img_3_1)))
        img_3_3 = self.convmax3_2(img_3_2)

        img_4_1 = self.batch_norm_4_1(self.relu(self.conv4_1(img_3_3)))
        img_4_2 = self.batch_norm_4_2(self.relu(self.conv4_2(img_4_1)))
        img_4_3 = self.convmax4_3(self.spatial_dropout(img_4_2))

        img_5_1 = self.batch_norm_5_1(self.relu(self.conv5_1(img_4_3)))
        img_5_2 = self.batch_norm_5_2(self.relu(self.conv5_2(img_5_1)))
        img_5_3 = self.spatial_dropout(img_5_2)

        # Decoder
        img_6_1 = self.batch_norm_up6_1(self.relu(self.upconv6_1(img_5_3)))
        img_6_1 = torch.cat((img_6_1, img_4_2), dim=1)
        img_6_1 = self.batch_norm_conv6_1(self.relu(self.conv6_1(img_6_1)))
        img_6_2 = self.batch_norm_6_2(self.relu(self.conv6_2(img_6_1)))

        img_7_1 = self.batch_norm_up7_1(self.relu(self.upconv7_1(img_6_2)))
        img_7_1 = torch.cat((img_7_1, img_3_2), dim=1)
        img_7_1 = self.batch_norm_conv7_1(self.relu(self.conv7_1(img_7_1)))
        img_7_2 = self.batch_norm_7_2(self.relu(self.conv7_2(img_7_1)))

        img_8_1 = self.batch_norm_up8_1(self.relu(self.upconv8_1(img_7_2)))
        img_8_1 = torch.cat((img_8_1, img_2_2), dim=1)
        img_8_1 = self.batch_norm_conv8_1(self.relu(self.conv8_1(img_8_1)))
        img_8_2 = self.batch_norm_8_2(self.relu(self.conv8_2(img_8_1)))

        img_9_1 = self.batch_norm_up9_1(self.relu(self.upconv9_1(img_8_2)))
        img_9_1 = torch.cat((img_9_1, img_1_2), dim=1)
        img_9_1 = self.batch_norm_conv9_1(self.relu(self.conv9_1(img_9_1)))
        img_9_2 = self.batch_norm_9_2(self.relu(self.conv9_2(img_9_1)))
        img_9_3 = self.batch_norm_9_3(self.relu(self.conv9_3(img_9_2)))
        img_9_4 = self.sigmoid(self.conv9_4(img_9_3))

        return img_9_4


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


