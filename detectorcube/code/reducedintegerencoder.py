import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class BackoneNode(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(BackoneNode, self).__init__()
        self.conv = nn.Conv2d(inchannels, outchannels, kernel_size=3, padding=1, bias=False)
        self.relu = nn.LeakyReLU(0.1)
        self.bn = nn.BatchNorm2d(outchannels)

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        out = self.bn(out)
        return out

class Backbone(nn.Module):
    def __init__(self, in_channels, out_channels,genome1,genome2,genome3):
        super(Backbone, self).__init__()

        # phase1
        self.phase1_1 = BackoneNode(in_channels, out_channels)
        self.phase1_2 = BackoneNode(in_channels, out_channels)
        self.phase1_3 = BackoneNode(in_channels, out_channels)
        self.phase1_4 = BackoneNode(in_channels, out_channels)

        

        # max pool and conv
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)


        # max pool and conv
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)



        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
       
               
    def integer_encoder(self,genome1,genome2,genome3,x):
        genome = genome1
        #print(genome)
        out3 = torch.zeros_like(x)           # Initialize with zeros
                                             # this ensures no breakage in flow of inputs
        out1 = self.phase1_1(x)

        out2 = x if genome[0] < 1 else self.phase1_2(out1)  

        inputs = [x]
        if genome[0] >= 2:
         inputs.append(out1)
        if genome[1] >= 1:
         inputs.append(out2)
        out3 = self.phase1_3(sum(inputs))

        if genome[0]<3 and genome[1]<2 and genome[2]<1:  # no node is connected to node4
            out4 = self.phase1_4(x)  
        elif  genome[0]>=3 and genome[1]>=2 and genome[2]>=1: #  123 all prev nodes are connected
            out4 = self.phase1_4(out1+out2+out3) 
        elif  genome[0]>=3 and genome[1]>=2 and genome[2]<1: # 1 and 2 nodes are connected
            out4 = self.phase1_4(out1+out2)  
        elif  genome[0]<3 and genome[1]>=2 and genome[2]>=1: # 2 and 3 are connected
            out4 = self.phase1_4(out2+out3) 
        elif  genome[0]>=3 and genome[1]<2 and genome[2]>=1: # 1 and 3 nodes are connected
            out4 = self.phase1_4(out1+out3)
        elif  genome[0]>=3 and genome[1]<2 and genome[2]<1: #  1  are connected
            out4 = self.phase1_4(out1) 
        elif  genome[0]<3 and genome[1]>=2 and genome[2]<1: # 2 nodes are connected
            out4 = self.phase1_4(out2)  
        elif  genome[0]<3 and genome[1]<2 and genome[2]>=1: # 3 are connected
            out4 = self.phase1_4(out3)     

            
        return out4,out3

# PLEASE OPTIMIZE HEHEHEHEHEH