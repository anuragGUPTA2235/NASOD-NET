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
        self.phase1_5 = BackoneNode(in_channels, out_channels)
        self.phase1_6 = BackoneNode(in_channels, out_channels)
        self.phase1_7 = BackoneNode(in_channels, out_channels)

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
        out6 = torch.zeros_like(x)           # Initialize with zeros
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

        if  genome[0]<4 and genome[1]<3 and genome[2]<2 and genome[3]<1: # 1234 all prev nodes are not connected
            out5 = self.phase1_5(x)
        elif  genome[0]>=4 and genome[1]>=3 and genome[2]>=2 and genome[3]>=1: # 1234 nodes are all connected
            out5 = self.phase1_5(out1+out2+out3+out4)  
        elif  genome[0]>=4 and genome[1]>=3 and genome[2]>=2 and genome[3]<1: # 123 are connected
            out5 = self.phase1_5(out1+out2+out3) 
        elif  genome[0] <4 and genome[1]>=3 and genome[2]>=2 and genome[3]>=1: # 234 nodes are connected
            out5 = self.phase1_5(out2+out3+out4)    
        elif  genome[0]>=4 and genome[1]>=3 and genome[2]<2 and genome[3]>=1: # 124 nodes are connected
            out5 = self.phase1_5(out1+out2+out4)    
        elif  genome[0]>=4 and genome[1]<3 and genome[2]>=2 and genome[3]>=1: # 134 nodes are connected
            out5 = self.phase1_5(out1+out3+out4)  
        elif  genome[0]>=4 and genome[1]<3 and genome[2]<2 and genome[3]<1: # 1 are connected
            out5 = self.phase1_5(out1) 
        elif  genome[0]<4 and genome[1]>=3 and genome[2]<2 and genome[3]<1: # 2 nodes are connected
            out5 = self.phase1_5(out2)    
        elif  genome[0]<4 and genome[1]<3 and genome[2]>=2 and genome[3]<1: # 3 nodes are connected
            out5 = self.phase1_5(out3)    
        elif  genome[0]<4 and genome[1]<3 and genome[2]<2 and genome[3]>=1: # 4 nodes are connected
            out5 = self.phase1_5(out4)      
        elif  genome[0]<4 and genome[1]>=3 and genome[2]<2 and genome[3]>=1: # 24 are connected
            out5 = self.phase1_5(out2+out4) 
        elif  genome[0] >=4 and genome[1]>=3 and genome[2]<2 and genome[3]<1: # 12 nodes are connected
            out5 = self.phase1_5(out1+out2)    
        elif  genome[0]<4 and genome[1]<3 and genome[2]>=2 and genome[3]>=1: # 34 nodes are connected
            out5 = self.phase1_5(out4+out3)   
        elif  genome[0]>=4 and genome[1]<3 and genome[2]<2 and genome[3]>=1: # 14 nodes are connected
            out5 = self.phase1_5(out1+out4)    
        elif  genome[0]<4 and genome[1]>=3 and genome[2]>=2 and genome[3]<1: # 23 nodes are connected
            out5 = self.phase1_5(out2+out3)    
        elif  genome[0]>=4 and genome[1]<3 and genome[2]>=2 and genome[3]<1: # 13 nodes are connected
            out5 = self.phase1_5(out1+out3) 

        #if  genome[0]<5 and genome[1]<4 and genome[2]<3 and genome[3]<2 and genome[4]<1: # 12345 all prev nodes are not connected
            #out6 = self.phase1_6(x)  
        if  genome[0]>=5 and genome[1]>=4 and genome[2]>=3 and genome[3]>=2 and genome[4]>=1: # 12345 all prev nodes are not connected
            out6 = self.phase1_6(out1+out2+out3+out4+out5)   
        elif  genome[0]>=5 and genome[1]<4 and genome[2]<3 and genome[3]<2  and genome[4]<1: # 1 are connected
            out6 = self.phase1_6(out1) 
        elif  genome[0]<5 and genome[1]>=4 and genome[2]<3 and genome[3]<2 and genome[4]<1: # 2 nodes are connected
            out6 = self.phase1_6(out2)    
        elif  genome[0]<5 and genome[1]<4 and genome[2]>=3 and genome[3]<2 and genome[4]<1: # 3 nodes are connected
            out6 = self.phase1_6(out3)    
        elif  genome[0]<5 and genome[1]<4 and genome[2]<3 and genome[3]>=2 and genome[4]<1: # 4 nodes are connected
            out6 = self.phase1_6(out4)                                                          
        elif  genome[0]<5 and genome[1]<4 and genome[2]<3 and genome[3]>=2 and genome[4]>=1: # 5 nodes are connected
            out6 = self.phase1_6(out5)
        elif  genome[0]>=5 and genome[1]<4 and genome[2]>=3 and genome[3]>=2 and genome[4]>=1: # 1345 are connected
            out6 = self.phase1_6(out1+out3+out4+out5) 
        elif  genome[0]>=5 and genome[1]>=4 and genome[2]>=3 and genome[3]<2 and genome[4]>=1: # 1235 nodes are connected
            out6 = self.phase1_6(out1+out2+out3+out5)    
        elif  genome[0]>=5 and genome[1]>=4 and genome[2]>=3 and genome[3]>=2 and genome[4]<1: # 1234 nodes are connected
            out6 = self.phase1_6(out1+out2+out3+out4)    
        elif  genome[0]<5 and genome[1]>=4 and genome[2]>=3 and genome[3]>=2 and genome[4]>=1: # 2345 nodes are connected
            out6 = self.phase1_6(out2+out3+out4+out5)                                                          
        elif  genome[0]>=5 and genome[1]>=4 and genome[2]<3 and genome[3]>=2 and genome[4]>=1: # 1245 nodes are connected
            out6 = self.phase1_6(out1+out2+out4+out5)   
##############################################################################
        elif  genome[0]<5 and genome[1]>=4 and genome[2]<3 and genome[3]>=2  and genome[4]>=1: # 245 are connected
            out6 = self.phase1_6(out2+out4+out5) 
        elif  genome[0]>=5 and genome[1]<4 and genome[2]>=3 and genome[3]<2 and genome[4]>=1: # 135 nodes are connected
            out6 = self.phase1_6(out1+out3+out5)    
        elif  genome[0]>=5 and genome[1]>=4 and genome[2]>=3 and genome[3]<2 and genome[4]<1: # 123 nodes are connected
            out6 = self.phase1_6(out1+out2+out3)    
        elif  genome[0]>=5 and genome[1]<4 and genome[2]>=3 and genome[3]>=2 and genome[4]<1: # 134 nodes are connected
            out6 = self.phase1_6(out1+out3+out4)                                                          
        elif  genome[0]<5 and genome[1]>=4 and genome[2]>=3 and genome[3]<2 and genome[4]>=1: # 235 nodes are connected
            out6 = self.phase1_6(out2+out3+out5)
        elif  genome[0]<5 and genome[1]<4 and genome[2]>=3 and genome[3]>=2 and genome[4]>=1: # 345 are connected
            out6 = self.phase1_6(out3+out4+out5) 
        elif  genome[0]<5 and genome[1]>=4 and genome[2]>=3 and genome[3]>=2 and genome[4]<1: # 234 nodes are connected
            out6 = self.phase1_6(out2+out3+out4)    
        elif  genome[0]>=5 and genome[1]>=4 and genome[2]<3 and genome[3]<2 and genome[4]>=1: # 125 nodes are connected
            out6 = self.phase1_6(out1+out2+out5)    
        elif  genome[0]>=5 and genome[1]<4 and genome[2]<3 and genome[3]>=2 and genome[4]>=1: # 145 nodes are connected
            out6 = self.phase1_6(out4+out1+out5)                                                          
        elif  genome[0]>=5 and genome[1]>=4 and genome[2]<3 and genome[3]>=2 and genome[4]<1: # 124 nodes are connected
            out6 = self.phase1_6(out1+out2+out4) 
####################################################################
        elif  genome[0]<5 and genome[1]>=4 and genome[2]<3 and genome[3]>=2  and genome[4]<1: # 24 are connected
            out6 = self.phase1_6(out2+out4) 
        elif  genome[0]>=5 and genome[1]>=4 and genome[2]<3 and genome[3]<2 and genome[4]<1: # 12 nodes are connected
            out6 = self.phase1_6(out2+out1)    
        elif  genome[0]<5 and genome[1]<4 and genome[2]>=3 and genome[3]>=2 and genome[4]<1: # 34 nodes are connected
            out6 = self.phase1_6(out3+out4)    
        elif  genome[0]>=5 and genome[1]<4 and genome[2]<3 and genome[3]<2 and genome[4]>=1: # 15 nodes are connected
            out6 = self.phase1_6(out1+out5)                                                          
        elif  genome[0]>=5 and genome[1]<4 and genome[2]<3 and genome[3]>=2 and genome[4]<1: # 14 nodes are connected
            out6 = self.phase1_6(out1+out4)
        elif  genome[0]<5 and genome[1]>=4 and genome[2]>=3 and genome[3]<2 and genome[4]<1: # 23 are connected
            out6 = self.phase1_6(out2+out3) 
        elif  genome[0]<5 and genome[1]<4 and genome[2]<3 and genome[3]>=2 and genome[4]>=1: # 45 nodes are connected
            out6 = self.phase1_6(out4+out5)    
        elif  genome[0]<5 and genome[1]>=4 and genome[2]<3 and genome[3]<2 and genome[4]>=1: # 25 nodes are connected
            out6 = self.phase1_6(out2+out5)    
        elif  genome[0]>=5 and genome[1]<4 and genome[2]>=3 and genome[3]<2 and genome[4]<1: # 13 nodes are connected
            out6 = self.phase1_6(out1+out3)                                                          
        elif  genome[0]<5 and genome[1]<4 and genome[2]>=3 and genome[3]<2 and genome[4]>=1: # 35 nodes are connected
            out6 = self.phase1_6(out3+out5) 
        ##node7  
#################################################################  
       
        if  genome[0]<6 and genome[1]<5 and genome[2]<4 and genome[3]<3  and genome[4]<2 and genome[5]<1: # 123456 are not connected
            out7 = self.phase1_7(x)    
        elif  genome[0]>=6 and genome[1]>=5 and genome[2]>=4 and genome[3]>=3  and genome[4]>=2 and genome[5]>=1: # 123456 are  connected
            out7 = self.phase1_7(out1+out2+out3+out4+out5+out6)  
        elif  genome[0]>=6 and genome[1]<5 and genome[2]<4 and genome[3]<3  and genome[4]<2 and genome[5]<1: # 1 are  connected
            out7 = self.phase1_7(out1)  
        elif  genome[0]<6 and genome[1]>=5 and genome[2]<4 and genome[3]<3  and genome[4]<2 and genome[5]<1: # 2 are  connected
            out7 = self.phase1_7(out2)                                                
        elif  genome[0]<6 and genome[1]<5 and genome[2]>=4 and genome[3]<3  and genome[4]<2 and genome[5]<1: # 3 are  connected
            out7 = self.phase1_7(out3)  
        elif  genome[0]<6 and genome[1]<5 and genome[2]<4 and genome[3]>=3  and genome[4]<2 and genome[5]<1: # 4 are  connected
            out7 = self.phase1_7(out4) 
        elif  genome[0]<6 and genome[1]<5 and genome[2]<4 and genome[3]<3  and genome[4]>=2 and genome[5]<1: # 5 are  connected
            out7 = self.phase1_7(out5)  
        elif  genome[0]<6 and genome[1]<5 and genome[2]<4 and genome[3]<3  and genome[4]<2 and genome[5]>=1: # 6 are  connected
            out7 = self.phase1_7(out6)


#########################################################

        elif  genome[0]<6 and genome[1]>=5 and genome[2]<4 and genome[3]>=3  and genome[4]<2 and genome[5]<1: # 24 are  connected
            out7 = self.phase1_7(out2+out4)    
        elif  genome[0]>=6 and genome[1]>=5 and genome[2]<4 and genome[3]<3  and genome[4]<2 and genome[5]<1: # 12 are  connected
            out7 = self.phase1_7(out1+out2)  
        elif  genome[0]<6 and genome[1]<5 and genome[2]>=4 and genome[3]>=3  and genome[4]<2 and genome[5]<1: # 34 are  connected
            out7 = self.phase1_7(out3+out4)  
        elif  genome[0]>=6 and genome[1]<5 and genome[2]<4 and genome[3]<3  and genome[4]>=2 and genome[5]<1: # 15 are  connected
            out7 = self.phase1_7(out1+out5)                                                
        elif  genome[0]<6 and genome[1]<5 and genome[2]<4 and genome[3]>=3  and genome[4]<2 and genome[5]>=1: # 46 are  connected
            out7 = self.phase1_7(out4+out6)  
        elif  genome[0]>=6 and genome[1]<5 and genome[2]<4 and genome[3]>=3  and genome[4]<2 and genome[5]<1: # 14 are  connected
            out7 = self.phase1_7(out4+out1) 
        elif  genome[0]<6 and genome[1]>=5 and genome[2]>=4 and genome[3]<3  and genome[4]<2 and genome[5]<1: # 23 are  connected
            out7 = self.phase1_7(out2+out3)  
        elif  genome[0]<6 and genome[1]<5 and genome[2]<4 and genome[3]>=3  and genome[4]>=2 and genome[5]<1: # 45 are  connected
            out7 = self.phase1_7(out4+out5)   
        elif  genome[0]<6 and genome[1]>=5 and genome[2]<4 and genome[3]<3  and genome[4]<2 and genome[5]>=1: # 26 are  connected
            out7 = self.phase1_7(out2+out6)  
        elif  genome[0]<6 and genome[1]<5 and genome[2]<4 and genome[3]<3  and genome[4]>=2 and genome[5]>=1: # 56 are  connected
            out7 = self.phase1_7(out5+out6) 
        elif  genome[0]<6 and genome[1]<5 and genome[2]>=4 and genome[3]<3  and genome[4]<2 and genome[5]>=1: # 36 are  connected
            out7 = self.phase1_7(out3+out6)  
        elif  genome[0]>=6 and genome[1]<5 and genome[2]<4 and genome[3]<3  and genome[4]<2 and genome[5]>=1: # 16 are  connected
            out7 = self.phase1_7(out6+out1)        
        elif  genome[0]<6 and genome[1]>=5 and genome[2]<4 and genome[3]<3  and genome[4]>=2 and genome[5]<1: # 25 are  connected
            out7 = self.phase1_7(out2+out5) 
        elif  genome[0]>=6 and genome[1]<5 and genome[2]>=4 and genome[3]<3  and genome[4]<2 and genome[5]<1: # 13 are  connected
            out7 = self.phase1_7(out1+out3)  
        elif  genome[0]<6 and genome[1]<5 and genome[2]>=4 and genome[3]<3  and genome[4]>=2 and genome[5]<1: # 35 are  connected
            out7 = self.phase1_7(out3+out5)   
            
###########################################
            
        elif  genome[0]<6 and genome[1]<5 and genome[2]>=4 and genome[3]<3  and genome[4]>=2 and genome[5]>=1: # 356 are  connected
            out7 = self.phase1_7(out3+out5+out6)  
        elif  genome[0]>=6 and genome[1]>=5 and genome[2]<4 and genome[3]<3  and genome[4]>=2 and genome[5]<1: # 125 are  connected
            out7 = self.phase1_7(out2+out1+out5) 
        elif  genome[0]<6 and genome[1]<5 and genome[2]<4 and genome[3]>=3  and genome[4]>=2 and genome[5]>=1: # 456 are  connected
            out7 = self.phase1_7(out4+out5+out6)  
        elif  genome[0]>=6 and genome[1]<5 and genome[2]>=4 and genome[3]<3  and genome[4]<2 and genome[5]>=1: # 136 are  connected
            out7 = self.phase1_7(out3+out6+out1)   
        elif  genome[0]>=6 and genome[1]<5 and genome[2]<4 and genome[3]>=3  and genome[4]>=2 and genome[5]<1: # 145 are  connected
            out7 = self.phase1_7(out1+out4+out5)  
        elif  genome[0]<6 and genome[1]>=5 and genome[2]<4 and genome[3]>=3  and genome[4]>=2 and genome[5]<1: # 245 are  connected
            out7 = self.phase1_7(out5+out4+out2) 
        elif  genome[0]<6 and genome[1]>=5 and genome[2]>=4 and genome[3]<3  and genome[4]<2 and genome[5]>=1: # 236 are  connected
            out7 = self.phase1_7(out3+out6+out2)  
        elif  genome[0]>=6 and genome[1]<5 and genome[2]<4 and genome[3]<3  and genome[4]>=2 and genome[5]>=1: # 156 are  connected
            out7 = self.phase1_7(out6+out1+out5)        
        elif  genome[0]>=6 and genome[1]>=5 and genome[2]<4 and genome[3]>=3  and genome[4]<2 and genome[5]<1: # 124 are  connected
            out7 = self.phase1_7(out2+out4+out1) 
        elif  genome[0]<6 and genome[1]>=5 and genome[2]<4 and genome[3]<3  and genome[4]>=2 and genome[5]>=1: # 256 are  connected
            out7 = self.phase1_7(out5+out2+out6)  
        elif  genome[0]<6 and genome[1]<5 and genome[2]>=4 and genome[3]>=3  and genome[4]<2 and genome[5]>=1: # 346 are  connected
            out7 = self.phase1_7(out3+out6+out4) 
        elif  genome[0]>=6 and genome[1]<5 and genome[2]>=4 and genome[3]<3  and genome[4]>=2 and genome[5]<1: # 135 are  connected
            out7 = self.phase1_7(out1+out5+out3)   
        elif  genome[0]<6 and genome[1]>=5 and genome[2]>=4 and genome[3]<3  and genome[4]>=2 and genome[5]<1: # 235 are  connected
            out7 = self.phase1_7(out2+out3+out5)  
        elif  genome[0]>=6 and genome[1]>=5 and genome[2]<4 and genome[3]<3  and genome[4]<2 and genome[5]>=1: # 126 are  connected
            out7 = self.phase1_7(out1+out2+out6) 
        elif  genome[0]>=6 and genome[1]>=5 and genome[2]>=4 and genome[3]<3  and genome[4]<2 and genome[5]<1: # 123 are  connected
            out7 = self.phase1_7(out3+out1+out2)  
        elif  genome[0]>=6 and genome[1]<5 and genome[2]<4 and genome[3]>=3  and genome[4]<2 and genome[5]>=1: # 146 are  connected
            out7 = self.phase1_7(out6+out1+out4)        
        elif  genome[0]>=6 and genome[1]<5 and genome[2]>=4 and genome[3]>=3  and genome[4]<2 and genome[5]<1: # 134 are  connected
            out7 = self.phase1_7(out1+out3+out4) 
        elif  genome[0]<6 and genome[1]<5 and genome[2]>=4 and genome[3]>=3  and genome[4]>=2 and genome[5]<1: # 345 are  connected
            out7 = self.phase1_7(out4+out3+out5)  
        elif  genome[0]<6 and genome[1]>=5 and genome[2]>=4 and genome[3]>=3  and genome[4]<2 and genome[5]<1: # 234 are  connected
            out7 = self.phase1_7(out3+out4+out2)
        elif  genome[0]<6 and genome[1]>=5 and genome[2]<4 and genome[3]>=3  and genome[4]<2 and genome[5]>=1: # 246 are  connected
            out7 = self.phase1_7(out4+out6+out2)    

#############
            
        elif  genome[0]>=6 and genome[1]<5 and genome[2]>=4 and genome[3]>=3  and genome[4]>=2 and genome[5]>=1: # 13456 are  connected
            out7 = self.phase1_7(out1+out3+out4+out5+out6)        
        elif  genome[0]>=6 and genome[1]>=5 and genome[2]>=4 and genome[3]>=3  and genome[4]<2 and genome[5]>=1: # 12346 are not connected
            out7 = self.phase1_7(out1+out2+out3+out4+out6) 
        elif  genome[0]>=6 and genome[1]>=5 and genome[2]>=4 and genome[3]<3  and genome[4]>=2 and genome[5]>=1: # 12356 are  connected
            out7 = self.phase1_7(out1+out2+out3+out5+out6)  
        elif  genome[0]>=6 and genome[1]>=5 and genome[2]>=4 and genome[3]<3  and genome[4]>=2 and genome[5]<1: # 12345 are  connected
            out7 = self.phase1_7(out1+out2+out3+out4+out5)
        elif  genome[0]<6 and genome[1]>=5 and genome[2]>=4 and genome[3]>=3  and genome[4]>=2 and genome[5]>=1: # 23456 are  connected
            out7 = self.phase1_7(out2+out3+out4+out5+out6)          
        elif  genome[0]>=6 and genome[1]>=5 and genome[2]<4 and genome[3]>=3  and genome[4]>=2 and genome[5]>=1: # 12456 are  connected
            out7 = self.phase1_7(out1+out2+out4+out5+out6)          


##############
        elif  genome[0]<6 and genome[1]<5 and genome[2]>=4 and genome[3]>=3  and genome[4]>=2 and genome[5]>=1: # 3456 are  connected
            out7 = self.phase1_7(out6+out3+out4+out5)        
        elif  genome[0]>=6 and genome[1]<5 and genome[2]>=4 and genome[3]>=3  and genome[4]>=2 and genome[5]<1: # 1345 are  connected
            out7 = self.phase1_7(out1+out3+out4+out5) 
        elif  genome[0]>=6 and genome[1]<5 and genome[2]<4 and genome[3]>=3  and genome[4]>=2 and genome[5]>=1: # 1456 are  connected
            out7 = self.phase1_7(out1+out4+out5+out6)  
        elif  genome[0]<6 and genome[1]>=5 and genome[2]<4 and genome[3]>=3  and genome[4]>=2 and genome[5]>=1: # 2456 are  connected
            out7 = self.phase1_7(out2+out4+out5+out6)
        elif  genome[0]>=6 and genome[1]>=5 and genome[2]>=4 and genome[3]<3  and genome[4]>=2 and genome[5]<1: # 1235 are  connected
            out7 = self.phase1_7(out1+out2+out3+out5)          
        elif  genome[0]>=6 and genome[1]>=5 and genome[2]>=4 and genome[3]>=3  and genome[4]<2 and genome[5]<1: # 1234 are  connected
            out7 = self.phase1_7(out1+out2+out3+out4)   
        elif  genome[0]<6 and genome[1]>=5 and genome[2]>=4 and genome[3]>=3  and genome[4]<2 and genome[5]>=1: # 2346 are  connected
            out7 = self.phase1_7(out6+out2+out4+out3)        
        elif  genome[0]<6 and genome[1]>=5 and genome[2]>=4 and genome[3]<3  and genome[4]>=2 and genome[5]>=1: # 2356 are  connected
            out7 = self.phase1_7(out2+out5+out3+out6) 
        elif  genome[0]>=6 and genome[1]>=5 and genome[2]<4 and genome[3]>=3  and genome[4]<2 and genome[5]>=1: # 1246 are  connected
            out7 = self.phase1_7(out1+out2+out4+out6)  
        elif  genome[0]<6 and genome[1]>=5 and genome[2]>=4 and genome[3]>=3  and genome[4]>=2 and genome[5]<1: # 2345 are  connected
            out7 = self.phase1_7(out2+out3+out4+out5)
        elif  genome[0]>=6 and genome[1]>=5 and genome[2]<4 and genome[3]<3  and genome[4]>=2 and genome[5]>=1: # 1256 are  connected
            out7 = self.phase1_7(out1+out2+out5+out6)          
        elif  genome[0]>=6 and genome[1]>=5 and genome[2]>=4 and genome[3]<3  and genome[4]<2 and genome[5]>=1: # 1236 are  connected
            out7 = self.phase1_7(out1+out2+out3+out6)  
        elif  genome[0]>=6 and genome[1]<5 and genome[2]>=4 and genome[3]>=3  and genome[4]<2 and genome[5]>=1: # 1346 are  connected
            out7 = self.phase1_7(out1+out3+out4+out6)
        elif  genome[0]>=6 and genome[1]>=5 and genome[2]<4 and genome[3]>=3  and genome[4]>=2 and genome[5]<1: # 1245 are  connected
            out7 = self.phase1_7(out1+out2+out4+out5)          
        elif  genome[0]>=6 and genome[1]<5 and genome[2]>=4 and genome[3]<3  and genome[4]>=2 and genome[5]>=1: # 1356 are  connected
            out7 = self.phase1_7(out1+out3+out5+out6)                                        
#########################
            
        return out7,out6

