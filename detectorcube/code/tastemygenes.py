import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from integer_encoder import Backbone,BackoneNode
from graphviz import Source
import os
from torchsummary import summary
import torch.nn as nn
from torchviz import make_dot

        

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
    def __init__(self, in_channels, out_channels, num_classes,genome1,genome2,genome3):
        super(Architecture, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_classes = num_classes
        self.genome1 = genome1
        self.genome2 = genome2
        self.genome3 = genome3
         
        
        # print(genome1)   
        
        
        # print("main evil")  
        
        

        # phase1
        self.phase1_1 = oneNode(in_channels, out_channels)

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
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=1)
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        '''
        print("this is the line")
        print(in_channels)
        print(out_channels)
        print(genome3)
        '''

    def forward(self,x):
########################################################################
        '''
        print("this is cinema hoey")
        print(self.in_channels)
        print(self.genome3)
        print("sdfmdvkdnvdbnvsdvcdsnvcdsbnvc nsbbv ")
        '''
        in_channels = self.in_channels
        out_channels = self.out_channels
        genome1 = self.genome1
        genome2 = self.genome2
        genome3 =self.genome3
        arch = Backbone(in_channels, out_channels,genome1,genome2,genome3).cuda()
        out7,out6 = arch.integer_encoder(genome1,genome2,genome3,x=x)
        out6 = self.phase1_1(x+out6)                            # as default node is connected to it!!!!
###############################################################
        out = self.conv1(out6+out7)   
        #out = self.avgpool1(out)
        out = self.conv2(out)
###############################################################
        arch = Backbone(in_channels, out_channels,genome1,genome2,genome3).cuda()
        out7,out6 = arch.integer_encoder(genome2,genome1,genome3,x=x)             
        out6 = self.phase1_1(x+out6)                            # as default node is connected to it!!!!
########################################################################
        out = self.conv3(out6+out7)
        #out = self.avgpool2(out)
        out = self.conv4(out)
######################################################################## 
        arch = Backbone(in_channels, out_channels,genome1,genome2,genome3).cuda()  
        out7,out6 = arch.integer_encoder(genome3,genome1,genome2,x=x)                   
        out6 = self.phase1_1    (x+out6) 
#######################################################################
        out = self.conv4(out6+out7)
        out = self.maxpool3(out)
        out = self.conv5(out)
#######################################################################
 
        return out


class objectNet(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes,genome1,genome2,genome3):
        super(objectNet, self).__init__()       
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_classes = num_classes
        self.genome1 = genome1
        self.genome2 = genome2
        self.genome3 = genome3
        #print(genome2)
        #rint("dnjksdcjksbcjkbsdkjvbsdjkcb snbvhjsdvnsd cnbsdnc smndbvcjsdc m,sdzbckj") 
        self.base_network = Architecture(in_channels, out_channels,num_classes,genome1,genome2,genome3)

  

    def forward(self, x):
        out = self.base_network(x)

        return out


class def_node(nn.Module):
    def __init__(self):
        super(def_node, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x).to(device)
        return x


class CombinedNet(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes,genome1,genome2,genome3):
        super(CombinedNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_classes = num_classes
        self.genome1 = genome1
        self.genome2 = genome2
        self.genome3 = genome3
        self.default_node = def_node()
        self.object_net = objectNet(in_channels, out_channels, num_classes,genome1,genome2,genome3)

    def forward(self, x):

     #   x = self.default_node(x)
   
        x = self.object_net(x)
  
        return x
    
'''
def main():
 genome1 = [6,5,4,3,2,1]
 genome2 = [5,5,4,3,2,1]
 genome3 = [6,5,4,1,1,1]
 device = "cuda"
 model = CombinedNet(128,128,91,genome1,genome2,genome3)
 #model = nn.DataParallel(model)
 model.to('cuda')
 x = torch.randn(1, 256, 128, 128).to("cuda")  # Change the shape according to your input size

 y = model(x)

 print(y.shape)
 input_size = (256, 128, 128)  # Example input size
 dummy_input = torch.randn(1, *input_size).to(device)

 #summary(model, input_size)

if __name__ == "__main__":

      main()
"""
graph = make_dot(y, params=dict(model.named_parameters()))
graph.render("model_graph")
"""
'''
