import torch
import torch.nn as nn
from collections import OrderedDict
from yolo_model.CSPdarknet import darknet53
from torchsummary import summary

def conv2d(filter_in, filter_out, kernel_size, stride=1):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))

#---------------------------------------------------#
#   SPP结构，利用不同大小的池化核进行池化
#   池化后堆叠
#---------------------------------------------------#
class SpatialPyramidPooling(nn.Module):
    def __init__(self, pool_sizes=[5, 9, 13]):
        super(SpatialPyramidPooling, self).__init__()

        # nn.MaxPool2d(kernel_size, stride, padding)
        self.maxpools = nn.ModuleList([nn.MaxPool2d(pool_size, 1, pool_size//2) for pool_size in pool_sizes])

    def forward(self, x):
        features = [maxpool(x) for maxpool in self.maxpools[::-1]]
        features = torch.cat(features + [x], dim=1)  # 13,13,4096

        return features

#---------------------------------------------------#
#   卷积 + 上采样
#---------------------------------------------------#
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            conv2d(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x,):
        x = self.upsample(x)
        return x

#---------------------------------------------------#
#   三次卷积块
#  conv2d (in_channels, out_channels, kernel_size, stride)
#---------------------------------------------------#
def make_three_conv(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
    )
    return m

#---------------------------------------------------#
#   五次卷积块
#---------------------------------------------------#
def make_five_conv(filters_list, in_filters):  # 256, 512, 512
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
    )
    return m

#---------------------------------------------------#
#   最后获得yolov4的输出
#---------------------------------------------------#
def yolo_head(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 3),
        nn.Conv2d(filters_list[0], filters_list[1], 1),
    )
    return m

#---------------------------------------------------#
#   Pretrained model
#---------------------------------------------------#
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        pretrained_net = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True)
        summary(pretrained_net, input_size=(3, 416, 416))
        self.stage52 = nn.Sequential(*list(pretrained_net.children())[0][:-12])
        # summary(self.stage52, input_size=(3, 416, 416))
        self.stage26 = nn.Sequential(*list(pretrained_net.children())[0][-12:-5])
        # summary(self.stage26, input_size=(32, 52, 52))
        self.stage13 = nn.Sequential(*list(pretrained_net.children())[0][-5:])
        summary(self.stage13, input_size=(96, 26, 26))

    def forward(self, x):
        out3 = self.stage52(x)
        out4 = self.stage26(out3)
        out5 = self.stage13(out4)
        return out3, out4, out5


#---------------------------------------------------#
#   yolo_body
#---------------------------------------------------#
class YoloBody(nn.Module):
    def __init__(self, num_anchors, num_classes):
        super(YoloBody, self).__init__()
        #  backbone
        self.backbone = darknet53(None)
        # MobileNet
        # self.backbone = Net()
        # summary(self.backbone, input_size=(3, 416, 416))

        self.conv1 = make_three_conv([512,1024],1024) # out, middle, in //  threelayer output channels change: 1024->512 512->1024 1024-512
        self.SPP = SpatialPyramidPooling() # in channel: 512, out channel:2048
        self.conv2 = make_three_conv([512,1024],2048) # out, middle, in //  threelayer output channels change: 2048->512 512->1024 1024-512

        self.upsample1 = Upsample(512,256)
        self.conv_for_P4 = conv2d(512,256,1)  # filter_in, filter_out, kernel_size
        self.make_five_conv1 = make_five_conv([256, 512],512)  # out, middle, in

        self.upsample2 = Upsample(256,128)
        self.conv_for_P3 = conv2d(256,128,1)
        self.make_five_conv2 = make_five_conv([128, 256],256)
        # 3*(5+num_classes)=3*(5+20)=3*(4+1+20)=75
        # 4+1+num_classes
        final_out_filter2 = num_anchors * (5 + num_classes)   # 9*(5+20)
        self.yolo_head3 = yolo_head([256, final_out_filter2],128) # middle_channels, out_channels, in_channels # feature map size remain unchange

        self.down_sample1 = conv2d(128,256,3,stride=2)
        self.make_five_conv3 = make_five_conv([256, 512],512)
        # 3*(5+num_classes)=3*(5+20)=3*(4+1+20)=75
        final_out_filter1 =  num_anchors * (5 + num_classes)
        self.yolo_head2 = yolo_head([512, final_out_filter1],256)


        self.down_sample2 = conv2d(256,512,3,stride=2)
        self.make_five_conv4 = make_five_conv([512, 1024],1024)
        # 3*(5+num_classes)=3*(5+20)=3*(4+1+20)=75
        final_out_filter0 =  num_anchors * (5 + num_classes)
        self.yolo_head1 = yolo_head([1024, final_out_filter0],512)


    def forward(self, x):
        #  backbone
        x2, x1, x0 = self.backbone(x)

        # SPP:
        P5 = self.conv1(x0)          # x0: 13,13,1024 ==> P5: 13, 13, 512
        P5 = self.SPP(P5)            # SPP P5: 13, 13, 512 ==> P5: 13,13,2048
        P5 = self.conv2(P5)          # P5: 13,13,2048 ==> P5: 13, 13, 512

        # Upsample 26by26 level
        P5_upsample = self.upsample1(P5)        # P5: 13, 13, 512 ==> P5_upsample: 26, 26, 256
        P4 = self.conv_for_P4(x1)               # x1: 26, 26, 512 ==> P4: 26, 26, 256
        P4 = torch.cat([P4,P5_upsample],axis=1) # P4 concatenation ==> P4: 26, 26, 512
        P4 = self.make_five_conv1(P4)           # P4: 26, 26, 512 ==> P4: 26, 26, 256

        #  Upsample 52by52 level
        P4_upsample = self.upsample2(P4)        # P4: 26, 26, 256 ==> P4_upsample: 52, 52, 128
        P3 = self.conv_for_P3(x2)               # x2: 52, 52, 256 ==> P3: 52, 52, 128
        P3 = torch.cat([P3,P4_upsample],axis=1) # P3 concatenation ==> P3: 52, 52, 256
        P3 = self.make_five_conv2(P3)           # P3: 52, 52, 256 ==> P3: 52, 52, 128

        P3_downsample = self.down_sample1(P3)       # P3: 52, 52, 128 =conv2d=> P3_downsample: 26, 26, 256
        P4 = torch.cat([P3_downsample,P4],axis=1)   # P4: 26, 26, 256 + P3_downsample: 26, 26, 256 ==> P4: 26, 26, 512
        P4 = self.make_five_conv3(P4)               # P4: 26, 26, 512 ==> P4: 26, 26, 256

        P4_downsample = self.down_sample2(P4)        # P4: 26, 26, 256 ==> P4_downsample: 13, 13, 512
        P5 = torch.cat([P4_downsample,P5],axis=1)    # P5: 13, 13, 512 + P4_downsample: 13, 13, 512 ==> P5: 13, 13, 1024
        P5 = self.make_five_conv4(P5)                # P5: 13, 13, 1024 ==> P5: 13, 13, 512

        out2 = self.yolo_head3(P3)          # P3: 52, 52, 128 =ConvBlock=> P3: 52, 52, 256 =1*1conv2d=> P3: 52, 52, num_anchors * (5 + num_classes)
        out1 = self.yolo_head2(P4)          # P4: 26, 26, 256 =ConvBlock=> P4: 26, 26, 512 =1*1conv2d=> P4: 26, 26, num_anchors * (5 + num_classes)
        out0 = self.yolo_head1(P5)          # P5: 13, 13, 512 =ConvBlock=> P5: 13, 13, 1024 =1*1conv2d=> P5: 13, 13, num_anchors * (5 + num_classes)

        return out0, out1, out2

