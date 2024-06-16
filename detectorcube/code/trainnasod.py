import torch as th
import torchvision.transforms as transforms
import torch.optim as opt
from torch.utils.data import DataLoader, WeightedRandomSampler
from dataset import VOC_Detection
from transforms import RandomScaleTranslate, Resize, RandomColorJitter, RandomHorizontalFlip, ToYOLOTensor
from nasodmodel import YOLOv1
from loss import YOLO_Loss
from tqdm import tqdm
from typing import List, Tuple, Dict
import torch.nn as nn

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group,destroy_process_group
import os
from evaluate import map_metric








# Model Hyperparameters
S = 7
B = 2
D = 448

# Loss Funcion Hyperparameters
L_COORD = 5.0
L_NOOBJ = 0.5

# Data Augmentation Hyperparameters
HUE = 0.1
SATURATION = 1.5
EXPOSURE = 1.5

RESIZE_PROB = 0.2
ZOOM_OUT_PROB = 0.4
ZOOM_IN_PROB = 0.4
JITTER = 0.2

# Data Loading Hyperparameters
BATCH = 20
SUBDIVISIONS = 8
NUM_WORKERS = 14
SHUFFLE = True
PIN_MEMORY = True
DROP_LAST = True

# Training Hyperparameters
MAX_EPOCHS = 3
INIT_LR = 0.0005
BURN_IN = 100
BURN_IN_POW = 2.
LR_SCHEDULE = [(750, 2.0),  # (step, scale)
               (1500, 2.0),
               (2250, 1.25),
               (3250, 1.60),
               (5500, 1.25),
               (15000, 0.8),
               (20000, 0.625),
               (25000, 0.8),
               (30000, 0.5),
               (35000, 0.5)]
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
global loss_arch


# VOC Dataset Directory
PASCAL_VOC_DIR_PATH = "/run/user/1001/projectredmark/"

# Compute Device (use a GPU if available)
DEVICE = 'cuda:0' 

# Checkpoint Hyperparameters
LOAD_MODEL = None  # 'pretrain', 'train', None
TRAINED_MODEL_WEIGHTS = "archive.pth"
PRETRAINED_WEIGHTS = "model_epoch_2.pt"
TRAINING_CHECKPOINT_PATH = "checkpoint.pt"


##########################################################################

class MultiStepScaleLR:

    def __init__(self,
                 optimizer: opt.SGD,
                 init_lr: float,
                 lr_schedule: List[Tuple[int, float]],
                 burn_in: int,
                 burn_in_pow: float) -> None:
    
        self.optimizer = optimizer
        self.steps, self.scales = zip(*lr_schedule)
        self.burn_in = burn_in
        self.init_lr = init_lr
        self.pow = burn_in_pow
        self.batch = 0
        self.next_step_ind = 0

    def step(self) -> None:
 
        self.batch += 1
        if self.batch < self.burn_in:
            self.optimizer.param_groups[0]['lr'] = self.init_lr * ((self.batch+1)/self.burn_in)**self.pow
        elif self.next_step_ind < len(self.steps) and self.batch == self.steps[self.next_step_ind]:
            self.optimizer.param_groups[0]['lr'] *= self.scales[self.next_step_ind]
            self.next_step_ind += 1




def train_epoch(train_loader: DataLoader,
                model: YOLOv1,
                optimizer: opt.SGD,
                criterion: YOLO_Loss,
                scheduler: MultiStepScaleLR,
                mini_batch: int) -> Tuple[float, int]:

    av_loss = 0.
    

    model.train()
    
    for x, y_gt in tqdm(train_loader):
        mini_batch += 1
        x, y_gt = x.to(DEVICE), y_gt.to(DEVICE)
        y_pred = model(x)
        loss = criterion(y_pred, y_gt) / SUBDIVISIONS
        loss.backward()
    
        
        

        if mini_batch == SUBDIVISIONS:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            mini_batch = 0

        av_loss += loss.item() * SUBDIVISIONS

        
        

    av_loss /= len(train_loader)
    return av_loss, mini_batch


def validate_epoch(val_loader: DataLoader,
                   model: YOLOv1,
                   criterion: YOLO_Loss) -> float:

    av_loss = 0.
    with th.no_grad():
        model.eval()
        for x, y_gt in val_loader:
            x, y_gt = x.to(DEVICE), y_gt.to(DEVICE)
            y_pred = model(x)
            loss = criterion(y_pred, y_gt)
            av_loss += loss.item()
            
            

    av_loss /= len(val_loader)
    return av_loss


def train(train_loader: DataLoader,
          test_loader: DataLoader,
          model: YOLOv1,
          optimizer: opt.SGD,
          criterion: YOLO_Loss,
          scheduler: MultiStepScaleLR,
          epoch: int,
          mini_batch: int,
          train_loss_history: List[float],
          test_loss_history: List[float]) -> None:



    save_epochs = list(range(1, 501, 10))  # List of epochs to save the model

    pbar = tqdm(total=MAX_EPOCHS, desc='Training Epoch', initial=epoch, unit='epoch', position=0, leave=True)
    if mini_batch == 0:
        optimizer.zero_grad()

    while epoch < MAX_EPOCHS:
        epoch += 1
    
        
       

        train_loss, mini_batch = train_epoch(train_loader, model, optimizer, criterion, scheduler, mini_batch)
        #test_loss = validate_epoch(test_loader, model, criterion)

        train_loss_history.append(train_loss)
        #test_loss_history.append(test_loss)

        print(train_loss)
        with open("/run/user/1001/projectredmark/new_nasod/bioex/loss.txt", "a") as file:
    # Write some content to the file
                            file.write(str(train_loss))
                            file.write("\n")
        loss_arch = train_loss
        #print(test_loss)
        
        #print(test_loss_history)
        

        



        if epoch == MAX_EPOCHS:
              th.save(model, TRAINED_MODEL_WEIGHTS)
        
              



def setup_train(genome1:list,genome2:list,genome3:list):

    

    model = YOLOv1(S=S,
                   B=B,
                   C=91,genome1=genome1,genome2=genome2,genome3=genome3)
                
                   

                    
    if th.cuda.device_count() > 1:
        print(f"Using {th.cuda.device_count()} GPUs")
        #model = th.nn.DataParallel(model)
    device = "cuda"
    model.to(device)    
    
       
    
 


    optimizer = opt.SGD(params=model.parameters(),
                        lr=INIT_LR * (1/BURN_IN)**BURN_IN_POW,
                        momentum=MOMENTUM,
                        weight_decay=WEIGHT_DECAY)

    scheduler = MultiStepScaleLR(optimizer,
                                 init_lr=INIT_LR,
                                 lr_schedule=LR_SCHEDULE,
                                 burn_in=BURN_IN,
                                 burn_in_pow=BURN_IN_POW)

    criterion = YOLO_Loss(S=S,
                          C=VOC_Detection.C,
                          B=B,
                          D=D,
                          L_coord=L_COORD,
                          L_noobj=L_NOOBJ).to("cuda")

    train_dataset = VOC_Detection(root_dir=PASCAL_VOC_DIR_PATH,
                                  split='train',
                                  transforms=transforms.Compose([
                                      RandomScaleTranslate(output_size=D,
                                                           jitter=JITTER,
                                                           resize_p=RESIZE_PROB,
                                                           zoom_out_p=ZOOM_OUT_PROB,
                                                           zoom_in_p=ZOOM_IN_PROB),
                                      RandomColorJitter(hue=HUE,
                                                        sat=SATURATION,
                                                        exp=EXPOSURE),
                                      RandomHorizontalFlip(p=0.5),
                                      ToYOLOTensor(S=S,
                                                   C=VOC_Detection.C,
                                                   normalize=[[0.4549, 0.4341, 0.4010],
                                                              [0.2703, 0.2672, 0.2808]])]))

    test_dataset = VOC_Detection(root_dir=PASCAL_VOC_DIR_PATH,
                                 split='train',
                                 transforms=transforms.Compose([
                                     Resize(output_size=D),
                                     ToYOLOTensor(S=S,
                                                  C=VOC_Detection.C,
                                                  normalize=[[0.4549, 0.4341, 0.4010],
                                                             [0.2703, 0.2672, 0.2808]])]))

    #train_sampler = DistributedSampler(dataset=train_dataset, shuffle=True)                                                             

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH // SUBDIVISIONS,
                              num_workers=NUM_WORKERS,
                              pin_memory=PIN_MEMORY,
                              shuffle=SHUFFLE,
                              drop_last=DROP_LAST)

    #test_sampler = DistributedSampler(dataset=test_dataset, shuffle=False)                           

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=BATCH // SUBDIVISIONS,
                             num_workers=NUM_WORKERS,
                             pin_memory=PIN_MEMORY)

    return train_loader, test_loader, model, optimizer, scheduler, criterion


def init_train(model: YOLOv1,
               optimizer: opt.SGD,
               scheduler: MultiStepScaleLR) -> Tuple[int, int, List[float], List[float]]:

    if LOAD_MODEL is None:
        epoch = 0
        mini_batch = 0
        train_loss_history = []
        test_loss_history = []

    else:
        assert 0

    return epoch, mini_batch, train_loss_history, test_loss_history


def make_archive(genome1:list,genome2:list,genome3:list):
    train_loader, test_loader, model, optimizer, scheduler, criterion  = setup_train(genome1,genome2,genome3)
    epoch, mini_batch, train_loss_hist, test_loss_hist = init_train(model, optimizer, scheduler)
    train(train_loader, test_loader, model, optimizer, criterion, scheduler,
          epoch, mini_batch,
          train_loss_hist, test_loss_hist)
    map_metric(genome1,genome2,genome3)      
          





