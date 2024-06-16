import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from reducedintegerencoder import Backbone,BackoneNode
from graphviz import Source
import os
from torchsummary import summary
import torch.nn as nn
from torchviz import make_dot

torch.manual_seed(199)

genome1 = [3,2,1]
genome2 = [3,2,1]
genome3 = [3,2,1]
device = "cuda"

class oneNode(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(oneNode, self).__init__()
        self.conv = nn.Conv2d(inchannels, outchannels, kernel_size=3, padding=1, bias=False)
        self.relu = nn.LeakyReLU(0.1)
        self.bn = nn.BatchNorm2d(outchannels)

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        out = self.bn(out)
        return out

class Architecture(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Architecture, self).__init__()

        # phase1
        self.phase1_1 = oneNode(in_channels, out_channels)
        self.phase1_2 = oneNode(in_channels, out_channels)
        self.phase1_3 = oneNode(in_channels, out_channels)
        self.phase1_4 = oneNode(in_channels, out_channels)

        # max pool and conv
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # max pool and conv
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        # maxpool and conv
        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
########################################################################
        in_channels,out_channels = 1024,1024
        arch = Backbone(in_channels, out_channels,genome1,genome2,genome3).to(device)
        out4,out3 = arch.integer_encoder(genome1,genome2,genome3,x=x)
        out3 = self.phase1_3(x+out3)                            # as default node is connected to it!!!!
###############################################################
        out = self.conv1(out3+out4)   
        out = self.avgpool1(out)
        out = self.conv2(out)
###############################################################
        arch = Backbone(in_channels, out_channels,genome1,genome2,genome3).to(device)
        out4,out3 = arch.integer_encoder(genome2,genome1,genome3,x=x)              
        out3 = self.phase1_3(x+out3)                            # as default node is connected to it!!!!
########################################################################
        out = self.conv3(out3+out4)
        out = self.avgpool2(out)
        out = self.conv4(out)
######################################################################## 
        arch = Backbone(in_channels, out_channels,genome1,genome2,genome3).to(device)     
        out4,out3 = arch.integer_encoder(genome3,genome1,genome2,x=x)                   
        out3 = self.phase1_3(x+out3) 
#######################################################################
        out = self.conv4(out3+out4)
        out = self.maxpool3(out)
        out = self.conv5(out)
#######################################################################
 
        return out


class objectNet(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes):
        super(objectNet, self).__init__()

        

        self.base_network = Architecture(in_channels=1024, out_channels=1024)

        self.flattening = nn.Flatten()

        self.trim1 = nn.Dropout(0.0)

        self.trim2 = nn.LeakyReLU(0.1)

        self.num_flat_features = None

    def forward(self, x):
        out = self.base_network(x)
        #print("base network our shape",out.shape)
        #out = self.flattening(out)

        #if self.num_flat_features is None:
         #   self.num_flat_features = out.size(1)
          #  self.fc1 = nn.Linear(self.num_flat_features, 496).to(device)
           # self.fc2 = nn.Linear(496, 2 * 2 * 12).to(device)

        #out = self.fc1(out)
        #out = self.trim1(out)
        #out = self.trim2(out)
        #out = self.fc2(out)
        return out


class def_node(nn.Module):
    def __init__(self):
        super(def_node, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=1024, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x).to(device)
        return x


class CombinedNet(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes):
        super(CombinedNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.default_node = def_node()
        self.object_net = objectNet(in_channels, out_channels, num_classes)

    def forward(self, x):
        #print(x.shape)
        #print("upto this it is correct")
        x = self.default_node(x)
        #print(x.shape)
        x = self.object_net(x)
        #print(x.shape)
        #print("hold my beer")
        return x
    

   
model = CombinedNet(1024,1024,20)
model = nn.DataParallel(model)
model = model.to('cuda')
summary(model,input_size=(3,128,128))
