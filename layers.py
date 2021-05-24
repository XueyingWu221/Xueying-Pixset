import torch
import torch.nn as nn



class BasicLayer(nn.Module):
    """ Basic convolution layer composed of a batch norm, activation then convolution operations.
        Works in both 2D and 3D.
    
        Args:
            in_channels - the number of input feature map channels
            out_channels - the number of output feature map channels
            kernel - the kernel for convolution in 3D format (depth, height, width), or 2D format (height, width).
            padding - if true, pads the input in order to conserve its dimensions. Similar as padding='same' for tensorflow.
            activation - activation function. Default is nn.ReLU()
            dropout - whether you want this BasicLayer to have an extra layer (after the convolution) of dropout.  
                        Specify the probability rate;  if set to None (by default), there will be no layer of dropout.
    """

    def __init__(self, in_channels:int, out_channels:int, kernel:tuple=(3,3,3), padding:bool=True, activation=nn.ReLU(), dropout=None):
        super(BasicLayer, self).__init__()
        self.dropout = dropout
        self.out_channels = out_channels

        if padding:
            padding = tuple(int((kernel[i]-1)/2) for i in range(len(kernel)))
        else:
            padding = 0
        # print("kernel = $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        # print(kernel)
        if len(kernel) == 3:
            # print("3D!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            self.bn = nn.BatchNorm3d(in_channels)
            self.conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, padding=padding, bias=False)
            self.activation = activation
            if self.dropout:
                self.drop = nn.Dropout3d(p=dropout)
        elif len(kernel) == 2:
            # print("222222222222222222222222222222222222222222222222222222222222222222222222222222222222")
            self.bn = nn.BatchNorm2d(in_channels)
            self.activation = activation
            # print("in_channels, out_channels = @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            # print(in_channels, out_channels)
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, padding=padding, bias=False)
            if self.dropout:
                self.drop = nn.Dropout2d(p=dropout)

    def forward(self, x):
        x = self.bn(x)
        x = self.conv(x)
        if self.activation:
            x = self.activation(x)
        if self.dropout:
            x = self.drop(x)
        return x



class DenseBlock(nn.Module):
    """ Dense block of fully convolutional layers. It combines 'n_layers' of basic_layer instances with a 
        given number of 'growth' (output channels). The input of each basic_layer 
        is formed from the outputs of all previous basic_layer.

        Args:
            in_channels - the number of input map features that enter the DenseBlock
            out_channels - the number fo out map features that will exit the DenseBlock 
            growth - the number of output features map after each basic_layer, except the last layer (=out_channels)
            n_layers - the total of layers in DenseBlock
            kernel - the size of the convolution kernel to be used in all basic_layers
            basic_layer - the type of basic layer you want as building blocks.
            dropout - None: there is no dropout layer in basic_layer after each convolution; set probability for dropout rate.  
                    ù) Note that, there is no dropout in first and last layers. (This is the rule).
        """

    def __init__(self, in_channels, out_channels='all', growth:int=12, n_layers:int=5, kernel:tuple=(7,3,3), basic_layer=BasicLayer, dropout=None):
        super(DenseBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.growth = growth
        self.n_layers = n_layers
        self.kernel = kernel
        self.basic_layer = basic_layer
        self.dropout = dropout
        self.layers = None

        self.fin_layer = False
        if self.out_channels is not 'all':
            self.fin_layer = True
            self.transi = None
        else:
            self.out_channels = self.in_channels + self.growth*self.n_layers

        self.construct_layers()

    def construct_layers(self):
        """This construct a list of BasicLayer with the right number of in_channels at each step.
            layers = [H_1, H_2, H_3, ..., H_nlayers]
             -> nn.ModuleList(layers)
        """
        layers = []

        for omega in range(self.n_layers): ## Here we don't use dropout for first and last layer
            in_chas = self.in_channels + self.growth*omega
            if omega == 0:
                layers.append(self.basic_layer(in_channels=in_chas, out_channels=self.growth, kernel=self.kernel, padding=True, activation=nn.ReLU(), dropout=None))
            elif omega == self.n_layers-1:
                layers.append(self.basic_layer(in_channels=in_chas, out_channels=self.growth, kernel=self.kernel, padding=True, activation=None, dropout=None))
            else:
                layers.append(self.basic_layer(in_channels=in_chas, out_channels=self.growth, kernel=self.kernel, padding=True, activation=nn.ReLU(), dropout=self.dropout))

        self.layers = nn.ModuleList(layers)

        if self.fin_layer:
            if len(self.kernel) == 2:
                self.transi = BasicLayer(in_channels=self.in_channels+self.growth*self.n_layers, out_channels=self.out_channels, 
                                         kernel=(1,1), padding=True, activation=None, dropout=None)
            elif len(self.kernel) == 3:
                self.transi = BasicLayer(in_channels=self.in_channels+self.growth*self.n_layers, out_channels=self.out_channels, 
                                         kernel=(1,1,1), padding=True, activation=None, dropout=None)

    def forward(self,x):

        for omega in range(self.n_layers):
            # print("omega = **************************************************************************************")
            # print(omega)
            # print("x.size = ")
            # print(x.size())
            # print("self.layers[omega](x).size() = ")
            # print(self.layers[omega](x).size())
            x = torch.cat((x, self.layers[omega](x)),1)

        if self.fin_layer:
            x = self.transi(x)

        return x

# DenseBlock2-------------------------------------------------------------------#
class DenseBlock2(nn.Module):
    """ Dense block of fully convolutional layers. It combines 'n_layers' of basic_layer instances with a
        given number of 'growth' (output channels). The input of each basic_layer
        is formed from the outputs of all previous basic_layer.

        Args:
            in_channels - the number of input map features that enter the DenseBlock
            out_channels - the number fo out map features that will exit the DenseBlock
            growth - the number of output features map after each basic_layer, except the last layer (=out_channels)
            n_layers - the total of layers in DenseBlock
            kernel - the size of the convolution kernel to be used in all basic_layers
            basic_layer - the type of basic layer you want as building blocks.
            dropout - None: there is no dropout layer in basic_layer after each convolution; set probability for dropout rate.
                    ù) Note that, there is no dropout in first and last layers. (This is the rule).
        """

    def __init__(self, in_channels, out_channels='all', growth:int=12, n_layers:int=5, kernel:tuple=(1,3,3), basic_layer=BasicLayer, dropout=None):
        super(DenseBlock2, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.growth = growth
        self.n_layers = n_layers
        self.kernel = kernel
        self.basic_layer = basic_layer
        self.dropout = dropout
        self.layers = None

        self.fin_layer = False
        if self.out_channels is not 'all':
            self.fin_layer = True
            self.transi = None
        else:
            self.out_channels = self.in_channels #+ self.growth*self.n_layers

        self.construct_layers()

    def construct_layers(self):
        """This construct a list of BasicLayer with the right number of in_channels at each step.
            layers = [H_1, H_2, H_3, ..., H_nlayers]
             -> nn.ModuleList(layers)
        """
        layers = []

        for omega in range(self.n_layers): ## Here we don't use dropout for first and last layer
            in_chas = self.in_channels #+ self.growth*omega
            if omega == 0:
                layers.append(self.basic_layer(in_channels=in_chas, out_channels=self.growth, kernel=self.kernel, padding=True, activation=nn.ReLU(), dropout=None))
            elif omega == self.n_layers-1:
                layers.append(self.basic_layer(in_channels=in_chas, out_channels=self.growth, kernel=self.kernel, padding=True, activation=None, dropout=None))
            else:
                layers.append(self.basic_layer(in_channels=in_chas, out_channels=self.growth, kernel=self.kernel, padding=True, activation=nn.ReLU(), dropout=self.dropout))

        self.layers = nn.ModuleList(layers)

        if self.fin_layer:
            if len(self.kernel) == 2:
                self.transi = BasicLayer(in_channels=self.in_channels, out_channels=self.out_channels,
                                         kernel=(1,1), padding=True, activation=None, dropout=None)
            elif len(self.kernel) == 3:
                self.transi = BasicLayer(in_channels=self.in_channels+self.growth*self.n_layers, out_channels=self.out_channels,
                                         kernel=(1,1,1), padding=True, activation=None, dropout=None)

    def forward(self,x):

        for omega in range(self.n_layers):
            # print("omega = **************************************************************************************")
            # print(omega)
            # print("x.size = ")
            # print(x.size())
            # print("self.layers[omega](x).size() = ")
            # print(self.layers[omega](x).size())
            x = x + self.layers[omega](x)

        if self.fin_layer:
            x = self.transi(x)

        return x


# RESNET-------------------------------------------------------------------------#
#
# class BasicBlock(nn.Module):    # layer = 18 / 34
#     expansion = 1
#
#     def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
#         super(BasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
#                                kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_channel)
#         self.relu = nn.ReLU()
#         self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
#                                kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channel)
#         self.downsample = downsample
#
#     def forward(self, x):
#         identity = x
#         if self.downsample is not None:
#             identity = self.downsample(x)
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         out += identity
#         out = self.relu(out)
#
#         return out
#
#
# class Bottleneck(nn.Module):    # layer = 50 / 101/ 152
#     expansion = 4
#
#     def __init__(self, in_channel, out_channel, stride=1, downsample=None,
#                  groups=1, width_per_group=64):
#         super(Bottleneck, self).__init__()
#
#         width = int(out_channel * (width_per_group / 64.)) * groups
#
#         self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
#                                kernel_size=1, stride=1, bias=False)  # squeeze channels
#         self.bn1 = nn.BatchNorm2d(width)
#         # -----------------------------------------
#         self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
#                                kernel_size=3, stride=stride, bias=False, padding=1)
#         self.bn2 = nn.BatchNorm2d(width)
#         # -----------------------------------------
#         self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion,
#                                kernel_size=1, stride=1, bias=False)  # unsqueeze channels
#         self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#
#     def forward(self, x):
#         identity = x
#         if self.downsample is not None:
#             identity = self.downsample(x)
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#
#         out = self.conv3(out)
#         out = self.bn3(out)
#
#         out += identity
#         out = self.relu(out)
#
#         return out
#
#
# class ResNet(nn.Module):
#
#     def __init__(self,
#                  block,
#                  blocks_num,
#                  in_channels,
#                  num_classes=1000,
#                  include_top=True,
#                  groups=1,
#                  width_per_group=64):
#         super(ResNet, self).__init__()
#         self.include_top = include_top
#         self.in_channel = in_channels
#         self.out_channels = 1031
#
#         self.transi = BasicLayer(in_channels=2048, out_channels=1031,
#                                          kernel=(1,1), padding=True, activation=None, dropout=None)
#
#         self.groups = groups
#         self.width_per_group = width_per_group
#
#         self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
#                                padding=3, bias=False)
#         self.bn1 = nn.BatchNorm2d(self.in_channel)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, 64, blocks_num[0])
#         self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
#         if self.include_top:
#             self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
#             self.fc = nn.Linear(512 * block.expansion, num_classes)
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#
#     def _make_layer(self, block, channel, block_num, stride=1):
#         downsample = None
#         if stride != 1 or self.in_channel != channel * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(channel * block.expansion))
#
#         layers = []
#         layers.append(block(self.in_channel,
#                             channel,
#                             downsample=downsample,
#                             stride=stride,
#                             groups=self.groups,
#                             width_per_group=self.width_per_group))
#         self.in_channel = channel * block.expansion
#
#         for _ in range(1, block_num):
#             layers.append(block(self.in_channel,
#                                 channel,
#                                 groups=self.groups,
#                                 width_per_group=self.width_per_group))
#
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         # print('size of input of resnet: ******************************************************')
#         # print(x.size())
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#
#         x = self.transi(x)
#
#         if self.include_top:
#             x = self.avgpool(x)
#             x = torch.flatten(x, 1)
#             x = self.fc(x)
#
#         return x
#
#
# def resnet34(num_classes=1000, include_top=True):
#     # https://download.pytorch.org/models/resnet34-333f7ec4.pth
#     return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)
#
#
# def resnet50(num_classes=1000, include_top=True):
#     # https://download.pytorch.org/models/resnet50-19c8e357.pth
#     return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)
#
#
# def resnet101(num_classes=1000, include_top=True):
#     # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
#     return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)
#
#
# def resnext50_32x4d(num_classes=1000, include_top=True):
#     # https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
#     groups = 32
#     width_per_group = 4
#     return ResNet(Bottleneck, [3, 4, 6, 3],
#                   num_classes=num_classes,
#                   include_top=include_top,
#                   groups=groups,
#                   width_per_group=width_per_group)
#
#
# def resnext101_32x8d(num_classes=1000, include_top=True):
#     # https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
#     groups = 32
#     width_per_group = 8
#     return ResNet(Bottleneck, [3, 4, 23, 3],
#                   num_classes=num_classes,
#                   include_top=include_top,
#                   groups=groups,
#                   width_per_group=width_per_group)
