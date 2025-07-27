import torch
from torch import nn
import numpy as np
import os
import torch.nn.functional as F
from networks.Emhsa import *





#将卷积层、BN、激活函数层顺序封装在一起
class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),   #不使用偏置项，归一化层后面会处理偏移和缩放
        )
        relu = nn.ReLU(inplace=True)    #括号内表示直接在输入上修改，减少内存占用

        bn = nn.BatchNorm2d(out_channels)  #use_batchnorm为真时才创建

        super(Conv2dReLU, self).__init__(conv, bn, relu)  ##


class ConvBNReLU(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride=1, padding=1, activation=True):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
            c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False
        )
        self.bn = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU()
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activation:
            x = self.relu(x)
        return x

#encoder包含两个3x3卷积和一个1x1卷积
class DoubleConv(nn.Module):

    def __init__(self, cin, cout):       #新加 channel, reduction, S
        super(DoubleConv, self).__init__()

        # self.channel=channel
        # self.reduction=reduction
        # self.S=S

        self.conv = nn.Sequential(
            ConvBNReLU(cin, cout, 3, 1, padding=1),
            ConvBNReLU(cout, cout, 3, stride=1, padding=1),

            ConvBNReLU(cout, cout, 3, stride=1, padding=1, activation=False)


        )


        self.conv1 = nn.Conv2d(cout, cout, 1)#可能为了融合前两个卷积的特征
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        # print(x.shape)      [1,3,224,224]
        x = self.conv(x)      #[1,64,224,224]

        # x = self.psa(x)       #此行新加

        h = x

        x = self.conv1(x)     #[1,64,224,224]
        x = self.bn(x)        #[1,64,224,224]
        x = h + x      #残差连接
        x = self.relu(x)      #[1,64,224,224]
        return x



class project(nn.Module):
    def __init__(self, in_dim, out_dim, stride, padding, activate, norm, last=False):
        super().__init__()
        self.out_dim = out_dim
        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.activate = activate()
        self.norm1 = norm(out_dim)
        self.last = last
        if not last:
            self.norm2 = norm(out_dim)

    def forward(self, x):#卷积、激活、归一化
        x = self.conv1(x)
        x = self.activate(x)
        # norm1
        Wh, Ww = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm1(x)
        x = x.transpose(1, 2).view(-1, self.out_dim, Wh, Ww)
        x = self.conv2(x)
        if not self.last:
            x = self.activate(x)
            # norm2
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm2(x)
            x = x.transpose(1, 2).view(-1, self.out_dim, Wh, Ww)
        return x



import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
class LayerNorm(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
class group_aggregation_bridge(nn.Module):
    def __init__(self, dim_xh, dim_xl, k_size=3, d_list=[1, 2, 5, 7], first=None):
        super().__init__()
        assert dim_xl % 4 == 0, "dim_xl must be divisible by 4 for grouping"
        group_size = dim_xl // 4

        self.pre_project = nn.Conv2d(dim_xh, dim_xl, 1)

        # Use LayerNorm with group_size channels (not dim_xl, since each chunk has group_size channels)
        self.g0 = nn.Sequential(
            nn.LayerNorm([group_size, first, first], eps=1e-5),  # Normalize each chunk with group_size channels
            nn.Conv2d(group_size, group_size, kernel_size=k_size, stride=1,
                      padding=(k_size + (k_size - 1) * (d_list[0] - 1)) // 2,
                      dilation=d_list[0], groups=group_size)
        )
        self.g1 = nn.Sequential(
            nn.LayerNorm([group_size, first, first], eps=1e-5),
            nn.Conv2d(group_size, group_size, kernel_size=k_size, stride=1,
                      padding=(k_size + (k_size - 1) * (d_list[1] - 1)) // 2,
                      dilation=d_list[1], groups=group_size)
        )
        self.g2 = nn.Sequential(
            nn.LayerNorm([group_size, first, first], eps=1e-5),
            nn.Conv2d(group_size, group_size, kernel_size=k_size, stride=1,
                      padding=(k_size + (k_size - 1) * (d_list[2] - 1)) // 2,
                      dilation=d_list[2], groups=group_size)
        )
        self.g3 = nn.Sequential(
            nn.LayerNorm([group_size, first, first], eps=1e-5),
            nn.Conv2d(group_size, group_size, kernel_size=k_size, stride=1,
                      padding=(k_size + (k_size - 1) * (d_list[3] - 1)) // 2,
                      dilation=d_list[3], groups=group_size)
        )

        self.tail_conv = nn.Sequential(
            nn.LayerNorm([dim_xl, first, first], eps=1e-5),
            nn.Conv2d(dim_xl, dim_xl, 1)
        )

    def forward(self, xh, xl):
        xh = self.pre_project(xh)
        xh = F.interpolate(xh, size=[xl.size(2), xl.size(3)], mode='bilinear', align_corners=True)

        # Chunk tensors along the channel dimension
        xh_chunks = torch.chunk(xh, 4, dim=1)
        xl_chunks = torch.chunk(xl, 4, dim=1)

        # Apply group operations
        x0 = self.g0(xh_chunks[0] + xl_chunks[0])
        x1 = self.g1(xh_chunks[1] + xl_chunks[1])
        x2 = self.g2(xh_chunks[2] + xl_chunks[2])
        x3 = self.g3(xh_chunks[3] + xl_chunks[3])

        # Concatenate the results and apply the tail convolution
        x = torch.cat((x0, x1, x2, x3), dim=1)
        x = self.tail_conv(x)

        return x

class UEncoder(nn.Module):

    def __init__(self):
        super(UEncoder, self).__init__()
        # self.in_channel=64
        self.res1 = DoubleConv(3, 64)      #channel=64, reduction=2, S=1
        # self.emhsa1 = NTB(in_channels=64, out_channels=64)    #新加
        self.pool1 = nn.MaxPool2d(2)

        self.res2 = DoubleConv(64, 128)   #channel=128, reduction=1, S=4
        # self.emhsa2 = NTB(in_channels=128, out_channels=128)   #新加
        self.pool2 = nn.MaxPool2d(2)

        self.res3 = DoubleConv(128, 256)   #channel=256, reduction=1, S=8
        # self.emhsa3 = NTB(in_channels=256, out_channels=256)   #新加
        self.pool3 = nn.MaxPool2d(2)

        self.res4 = DoubleConv(256, 512)   #channel=512, reduction=1, S=16
        # self.emhsa4 = NTB(in_channels=512, out_channels=512)    #新加
        self.pool4 = nn.MaxPool2d(2)

        self.res5 = DoubleConv(512, 1024)   #channel=1024, reduction=1, S=32
        # self.emhsa5 = NTB(in_channels=1024, out_channels=1024)    #新加
        # self.pool5 = nn.MaxPool2d(2)


    def forward(self, x):
        features = []  #接收中间特征图，以便解码器进行上采样和融合特征
        x = self.res1(x)

        # x=self.emhsa1(x)      #新加
        features.append(x)  # (224, 224, 64)
        # print(x)
        x = self.pool1(x)  # (112, 112, 64)

        x = self.res2(x)
        # x=self.emhsa2(x)    #新加
        features.append(x)  # (112, 112, 128)
        x = self.pool2(x)  # (56, 56, 128)

        x = self.res3(x)
        # x=self.emhsa3(x)    #新加
        features.append(x)  # (56, 56, 256)
        x = self.pool3(x)  # (28, 28, 256)

        x = self.res4(x)
        # x=self.emhsa4(x)    #新加
        features.append(x)  # (28, 28, 512)
        x = self.pool4(x)  # (14, 14, 512)

        x = self.res5(x)
        # x=self.emhsa5(x)    #新加
        features.append(x)  # (14, 14, 1024)
        # x = self.pool5(x)  # (7, 7, 1024)
        # features.append(x)
        return features

class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2) #双线性上采样层，参数表示将长和宽都放大两倍

    def forward(self, x, skip=None):
        if skip is not None:
            x = self.up(x)  #输入先进行上采样

        if skip is not None:
            x = torch.cat([x, skip], dim=1)  #将特征图与skip在通道维度上拼接
        x = self.conv1(x)
        x = self.conv2(x)
        return x

#分割头
class SegmentationHead(nn.Sequential):
    #out_channels是目标类别数或特征图的数量，上采样因子为1，表示不进行上采样
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()#用于决定是否包含上采样层
        super().__init__(conv2d, upsampling)

#输出GFFM跳跃之后的特征
class ParallEncoder(nn.Module):   #得到的是GFFM跳跃连接之后的分层特征
    def __init__(self):
        super(ParallEncoder, self).__init__()
        self.Encoder1 = UEncoder()

        self.GAB4=group_aggregation_bridge(1024, 512, first=28)
        self.GAB3=group_aggregation_bridge(512, 256, first=56)
        self.GAB2=group_aggregation_bridge(256, 128, first=112)
        self.GAB1=group_aggregation_bridge(128, 64, first=224)


        self.emgsa=NTB(in_channels=1024,out_channels=1024)    #新加，消融实验2所做   没有NGTB



    def forward(self, x):
        skips = []
        features = self.Encoder1(x)
        features4 = features[4]    #[1,1024,14,14]


        features4 = self.emgsa(features4)   #新加[1,1024,14,14]   消融实验2所做  没有NGTB


        skips.append(features4)   #瓶颈处

        t4 = self.GAB4(features[4], features[3])     #features[4]效果最好的方式   TUE TU2ES
        # t4 = self.GAB4(features4, features[3])     #features4   TUES


        skips.append(t4)      #第一个GFFM
        t3 = self.GAB3(t4, features[2])
        skips.append(t3)      #第二个GFFM
        t2 = self.GAB2(t3, features[1])

        skips.append(t2)      #第三个GFFM
        t1 = self.GAB1(t2, features[0])
        skips.append(t1)      #第四个GFFM

        return skips




class Model(nn.Module):
    def __init__(self, n_classes=9):
        super(Model, self).__init__()
        self.p_encoder = ParallEncoder()  #设置2，4
        # self.Encoder1 = UEncoder()      #设置3，1      最好情况去掉这一行



        self.decoder5 = DecoderBlock(in_channels=1536,out_channels=512)
        self.decoder4 = DecoderBlock(in_channels=768,out_channels=256)
        self.decoder3 = DecoderBlock(in_channels=384,out_channels=128)
        self.decoder2 = DecoderBlock(in_channels=192,out_channels=64)

        # self.emgsa = NTB(in_channels=1024,out_channels=1024)    #设置3
        # self.emgsa5=NTB(in_channels=512,out_channels=512)     #新加的    TU2ES
        # self.emgsa4=NTB(in_channels=256,out_channels=256)     #新加的    TU3ES
        # self.emgsa3=NTB(in_channels=128,out_channels=128)     #新加的    TU4ES
        # self.emgsa2=NTB(in_channels=64,out_channels=64)     #新加的    TU5ES



        self.segmentation_head2 = SegmentationHead(
            in_channels=256,
            out_channels=n_classes,
            kernel_size=1,
        )
        self.segmentation_head3 = SegmentationHead(
            in_channels=128,
            out_channels=n_classes,
            kernel_size=1,
        )
        self.segmentation_head4 = SegmentationHead(
            in_channels=64,
            out_channels=n_classes,
            kernel_size=1,
        )
        self.segmentation_head5 = SegmentationHead(
            in_channels=64,
            out_channels=n_classes,
            kernel_size=1,
        )
        self.decoder_final = DecoderBlock(in_channels=64, out_channels=64)




    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        encoder_skips = self.p_encoder(x)   #设置2,4
        # features = self.Encoder1(x)    #设置1,3     最好情况去掉这一行

        # features[4] = self.emgsa(features[4]) #设置3





        # f5=self.decoder5(encoder_skips[0])    #[,512,14,14]
        # f5 = F.interpolate(f5, scale_factor=2, mode='bilinear', align_corners=True)
        # out5=torch.add(f5,encoder_skips[1])

        # encoder_skips[1]=self.emgsa5(encoder_skips[1])    #新加的  TU2ES


        f5=self.decoder5(encoder_skips[0],encoder_skips[1])  #设置2,4    消融个数1
        # f5=self.decoder5(features[4],features[3])    #设置3，1

        # f4=self.decoder4(out5)
        # f4 = F.interpolate(f4, scale_factor=2, mode='bilinear', align_corners=True)
        # out4=torch.add(f4,encoder_skips[2])

        # encoder_skips[2]=self.emgsa4(encoder_skips[2])    #新加的  TU3ES


        f4=self.decoder4(f5,encoder_skips[2])     #设置2,4   消融个数2
        # f4=self.decoder4(f5,features[2])           #设置1,3


        # f3=self.decoder3(out4)
        # f3 = F.interpolate(f3, scale_factor=2, mode='bilinear', align_corners=True)
        # out3=torch.add(f3,encoder_skips[3])

        # encoder_skips[3]=self.emgsa3(encoder_skips[3])    #新加的  TU4ES

        f3=self.decoder3(f4,encoder_skips[3])      #设置2,4      消融个数3
        # f3= self.decoder3(f4,features[1])           #设置1,3


        # f2=self.decoder2(out3)
        # f2 = F.interpolate(f2, scale_factor=2, mode='bilinear', align_corners=True)
        # out2=torch.add(f2,encoder_skips[4])

        # encoder_skips[4]=self.emgsa2(encoder_skips[4])    #新加的  TU5ES

        f2=self.decoder2(f3,encoder_skips[4])     #设置2,4
        # f2=self.decoder2(f3,features[0])          #设置1,3


        x_final = self.decoder_final(f2,None)

        # x2=self.segmentation_head2(out4)
        x2=self.segmentation_head2(f4)
        # x3=self.segmentation_head3(out3)
        x3=self.segmentation_head3(f3)
        # x4=self.segmentation_head4(out2)
        x4=self.segmentation_head4(f2)
        # x5=self.segmentation_head5(out2)
        logits=self.segmentation_head5(x_final)

        x2=F.interpolate(x2,scale_factor=4,mode='bilinear')
        x3=F.interpolate(x3,scale_factor=2,mode='bilinear')
        x4=F.interpolate(x4,scale_factor=1,mode='bilinear')

        return x2,x3,x4,logits
        # return logits





        # encoder_skips = self.p_encoder(x)

        # x1_up = self.decoder1(encoder_skips[-1], encoder_skips[-2])
        # x2_up = self.decoder2(x1_up, encoder_skips[-3])
        # x3_up = self.decoder3(x2_up, encoder_skips[-4])
        # x4_up = self.decoder4(x3_up, encoder_skips[-5])
        # x_final = self.decoder_final(x4_up, None)
        #
        #
        # x2_up = self.segmentation_head2(x2_up)
        # x3_up = self.segmentation_head3(x3_up)
        # x4_up = self.segmentation_head4(x4_up)
        # logits = self.segmentation_head5(x_final)
        #
        #
        # x2_up = F.interpolate(x2_up, scale_factor=8, mode='bilinear')
        # x3_up = F.interpolate(x3_up, scale_factor=4, mode='bilinear')
        # x4_up = F.interpolate(x4_up, scale_factor=2, mode='bilinear')
        #
        #
        #
        #
        # return x2_up,x3_up,x4_up,logits

if __name__ == '__main__':
    # ------------------------------------------------------------------------------------------------------------------


    model = Model()
    from fvcore.nn import FlopCountAnalysis
    input = torch.randn((1, 3, 224, 224))
    x2_up, x3_up, x4_up, logits = model(input)
    print(x2_up.shape,x3_up.shape,x4_up.shape,logits.shape)  #输出全为[1, 9, 224, 224]
    # flops = FlopCountAnalysis(model, input)
    # print("Total Flops: {}G".format(flops.total()*1e-9))
    print('# generator parameters:{}M'.format(1.0 * sum(param.numel() for param in model.parameters() if param.requires_grad)/1000000))

