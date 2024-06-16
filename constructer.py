"""

██╗     ██╗ ██████╗ ██╗  ██╗████████╗    ███╗   ██╗ █████╗ ███████╗ ██████╗ ██████╗ 
██║     ██║██╔════╝ ██║  ██║╚══██╔══╝    ████╗  ██║██╔══██╗██╔════╝██╔═══██╗██╔══██╗
██║     ██║██║  ███╗███████║   ██║       ██╔██╗ ██║███████║███████╗██║   ██║██║  ██║
██║     ██║██║   ██║██╔══██║   ██║       ██║╚██╗██║██╔══██║╚════██║██║   ██║██║  ██║
███████╗██║╚██████╔╝██║  ██║   ██║       ██║ ╚████║██║  ██║███████║╚██████╔╝██████╔╝
╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝       ╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚═════╝ 
                                                                                    
██████╗  ██████╗ ██████╗ ██████╗                                                    
╚════██╗██╔═████╗╚════██╗╚════██╗                                                   
 █████╔╝██║██╔██║ █████╔╝ █████╔╝                                                   
██╔═══╝ ████╔╝██║██╔═══╝  ╚═══██╗                                                   
███████╗╚██████╔╝███████╗██████╔╝                                                   
╚══════╝ ╚═════╝ ╚══════╝╚═════╝                                                    
                                                                                    
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from ret_encoder import Backbone,BackoneNode
from thop import profile
from torchviz import make_dot
from graphviz import Source
from thop import clever_format 
import os
from torchsummary import summary
import random
torch.manual_seed(199)

genome1 = []
genome2 = []
genome3 = []
device = "cuda"
childgene = []

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
        self.in_channels = in_channels
        self.out_channels = out_channels

        # phase1
        self.phase1_1 = oneNode(in_channels, out_channels)
        self.phase1_2 = oneNode(in_channels, out_channels)
        self.phase1_3 = oneNode(in_channels, out_channels)
        self.phase1_4 = oneNode(in_channels, out_channels)
        self.phase1_5 = oneNode(in_channels, out_channels)
        self.phase1_6 = oneNode(in_channels, out_channels)
        self.phase1_7 = oneNode(in_channels, out_channels)

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
        arch = Backbone(in_channels, out_channels,genome1,genome2,genome3).to(device)
        out7,out6 = arch.integer_encoder(genome1,genome2,genome3,x=x)
        out6 = self.phase1_6(x+out6)                            # as default node is connected to it!!!!
###############################################################
        out = self.conv1(out6+out7)   
        out = self.avgpool1(out)
        out = self.conv2(out)
###############################################################
        arch = Backbone(in_channels, out_channels,genome1,genome2,genome3).to(device)
        out7,out6 = arch.integer_encoder(genome2,genome1,genome3,x=x)              
        out6 = self.phase1_6(x+out6)                            # as default node is connected to it!!!!
########################################################################
        out = self.conv3(out6+out7)
        out = self.avgpool2(out)
        out = self.conv4(out)
######################################################################## 
        arch = Backbone(in_channels, out_channels,genome1,genome2,genome3).to(device)     
        out7,out6 = arch.integer_encoder(genome3,genome1,genome2,x=x)                   
        out6 = self.phase1_6(x+out6) 
#######################################################################
        out = self.conv4(out6+out7)
        out = self.maxpool3(out)
        out = self.conv5(out)
#######################################################################
 
        return out


class objectNet(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes,flat_neurons):
        super(objectNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_classes = num_classes
        self.flat_neurons = flat_neurons

        

        self.base_network = Architecture(in_channels, out_channels)

        self.flattening = nn.Flatten()

        self.trim1 = nn.Dropout(0.0)

        self.trim2 = nn.LeakyReLU(0.1)

        self.num_flat_features = None

    def forward(self, x):
        out = self.base_network(x)
        out = self.flattening(out)

        if self.num_flat_features is None:
            self.num_flat_features = out.size(1)
            self.fc1 = nn.Linear(self.num_flat_features, self.flat_neurons).to(device)
            self.fc2 = nn.Linear(self.flat_neurons, 7 * 7 * 30).to(device)

        out = self.fc1(out)
        out = self.trim1(out)
        out = self.trim2(out)
        out = self.fc2(out)

        return out


class def_node(nn.Module):
    def __init__(self):
        super(def_node, self).__init__()
        self.conv1 = nn.Conv2d(3, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x).to(device)
        return x


class CombinedNet(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes,flat_neurons):
        super(CombinedNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.flat_neurons = flat_neurons
        self.default_node = def_node()
        self.object_net = objectNet(in_channels,out_channels, num_classes,flat_neurons)

    def forward(self, x):
        x = self.default_node(x)
        x = self.object_net(x)
        return x


"""
def main():
 global genome1,genome2,genome3
 global in_channels,out_channels,flat_neurons
 genes = return_genes()
 #print(genes)
 for item in range(5):
    num_classes = 20
    print(genes[item][0])
    genome1, genome2, genome3 = genes[item][0], genes[item][1], genes[item][2]
    out_channels = random.randint(50,500)
    in_channels = out_channels
    flat_neurons = random.randint(50,500)
    input_data = torch.randn(1,3,128,128).to(device)
    combined_net = CombinedNet(in_channels, out_channels, num_classes,flat_neurons).to(device)
    print(f"Iteration {item+1}: Individual of Population Metrics are saved in BIOEX folder")
    output_data = combined_net(input_data.to(device))
    print(output_data.shape)
    output_data = output_data.reshape(-1, 7, 7, 30)
    #    print(output_data)
    print("Gene pool ",genome1,genome2,genome3)
    print("hyperparameters channels flatneurons : ",out_channels,flat_neurons)
    folder_name = f"bioex/architecture{item+1}"
    folder_path = os.path.join(os.getcwd(), folder_name)
    graph = make_dot(combined_net(input_data), params=dict(combined_net.named_parameters()))
    graph.render(os.path.join(folder_path,"computationgraph"), format="pdf")
    params=dict(combined_net.named_parameters())
    #print("arch parameters",params)
   
    for param in combined_net.parameters():
        params=params+1
        print("number of params",params)
     #break 
  
    print(summary(combined_net, input_size=((3, 128, 128))))  
    flops, params = profile(combined_net, inputs=(input_data,))
    flops_formatted = clever_format([flops], "%.3f")
    print("\033[91m" + f"FLOPs: {flops_formatted}" + "\033[0m")

    #print(f"Number of parameters: {params}") 
    #print("complete")
    #print(output_data.shape)

if __name__ == "__main__":
    main()
""" 


import logging
import sys
from contextlib import contextmanager

# Suppress all INFO logging
logging.getLogger().setLevel(logging.WARNING)

@contextmanager
def suppress_stdout():
    with open('/dev/null', 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout   


def flops_returner(genome):
 
 with suppress_stdout():
  global genome1,genome2,genome3
  global in_channels,out_channels,flat_neurons
  num_classes = 20
  genome1, genome2, genome3 = list((genome[:6])),list((genome[6:12])), list((genome[12:18]))
  genome1 = [int(items) for items in genome1]
  genome2 = [int(items) for items in genome2]
  genome3 = [int(items) for items in genome3]
  out_channels = int(genome[18:21])
  in_channels = out_channels
  flat_neurons = int(genome[21:24])
  input_data = torch.randn(1,3,128,128).to(device)
  combined_net = CombinedNet(in_channels, out_channels, num_classes,flat_neurons).to(device) 
  output_data = combined_net(input_data.to(device))
  flops, _ = profile(combined_net, inputs=(input_data,))
  flops_formatted = clever_format([flops], "%.3f")
  return flops_formatted








        
    


