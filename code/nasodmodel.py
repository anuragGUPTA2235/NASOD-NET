import torch as th
import torch.nn as nn
import torch.nn.functional as F
from math import ceil, floor
from typing import Optional, List, Tuple, Union
from torchsummary import summary
import torch.nn as nn
from tastemygenes import CombinedNet 
from torchviz import make_dot

class LocallyConnected2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 input_h: int,
                 input_w: int,
                 kernel_size: int,
                 stride: Optional[int] = 1,
                 padding: Optional[int] = 0) -> None:

        super(LocallyConnected2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.output_h = floor((input_h + 2 * padding - kernel_size) / stride + 1)
        self.output_w = floor((input_w + 2 * padding - kernel_size) / stride + 1)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(th.randn(1, self.in_channels, self.out_channels,
                                            self.output_h, self.output_w,
                                            self.kernel_size, self.kernel_size))
        self.bias = nn.Parameter(th.randn(1, self.out_channels, self.output_h, self.output_w))

    def forward(self, x: th.Tensor) -> th.Tensor:   
        x = F.pad(x, (self.padding,) * 4)
        windows = x.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)[:, :, None, ...]
        y = th.sum(self.weight * windows, dim=[1, 5, 6]) + self.bias
        return y

class YOLOv1(nn.Module):
    def __init__(self, S: int, B: int, C: int, mode: Optional[str] = 'detection') -> None:
        super(YOLOv1, self).__init__()
        self.S = S
        self.B = B
        self.C = 91
        self.mode = mode
        self.preblock1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.preblock2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.preblock3 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.preblock4 = nn.Conv2d(192, 256, kernel_size=3)
        self.lastblock7 = nn.Conv2d(256, 512, kernel_size=2)
        self.lastblock8 = nn.Conv2d(512, 1024, kernel_size=2)
        self.ultblock = nn.Conv2d(1024,1024,kernel_size=2)
        genome1 = [6,5,4,3,2,1]
        genome2 = [6,5,4,3,0,0]
        genome3 = [6,5,4,3,2,1]           
        self.backbone = CombinedNet(256,256,91,genome1,genome2,genome3)
        if mode == 'detection':
            in_channels=1024
            detection_fc_modules = nn.Sequential(LocallyConnected2d(in_channels, 256, 7, 7, 3, 1, 1),
                                                 nn.LeakyReLU(0.1),
                                                 nn.Flatten(),
                                                 nn.Dropout(p=0.5),
                                                 nn.Linear(256 * 7 * 7, S * S * (C + B * 5)))
            nn.init.kaiming_normal_(detection_fc_modules[0].weight, a=0.1, mode='fan_out')
            nn.init.zeros_(detection_fc_modules[0].bias)
            self.detection_head = nn.Sequential(detection_fc_modules)
            self.forward = self._forward_detection

    def _forward_detection(self, x: th.Tensor) -> th.Tensor:
        x = self.preblock1(x)
        x = self.preblock2(x)#max
        x = self.preblock3(x)
        x = self.preblock2(x)#max
        x = self.preblock4(x)  
        x = self.preblock2(x)#max 
        #x = self.preblock5(x)
        #x = self.preblock6(x)    
        x = self.backbone(x)
        x = self.lastblock7(x)
        x = self.preblock2(x)
        x = self.lastblock8(x)
        x = self.ultblock(x)
        x = self.ultblock(x)
        x = F.max_pool2d(x, kernel_size=5, stride=4)
        x = x[:, :, 0:14, 0:14]
        x = F.interpolate(x, size=(7, 7), mode='bilinear', align_corners=False)
        x = self.detection_head(x)
        y = x.reshape(x.shape[0], self.S, self.S, self.C + self.B * 5)
        return y
    
"""
model = YOLOv1(S=7, B=2, C=91)
model = nn.DataParallel(model)
model = model.cuda()
#summary(model,input_size=(3,448,448))
x = th.randn(1, 3, 448, 448).cuda()# Change the shape according to your input size
# Visualize the model graph
y = model(x)
print(y.shape)
"""

"""
graph = make_dot(y, params=dict(model.named_parameters()))
graph.render("model_graph")
"""

