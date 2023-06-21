# -*- coding:utf-8 -*-
import sys
sys.path.append('F:/Code/RANet-Submitted-to-IEEE-JSTARS-main')
import os, argparse, time

import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torch.nn.parallel.data_parallel import data_parallel
import torchvision
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from dataset import *
from RANet import *
# from networks.lr_schedule import *
from metric import *
from plot import *
from config1 import config

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def inference():
    # model
    # load checkpoint
    # device = torch.device("cuda:1")
    # print(device)
    # model = torch.load('/data3/qiaoxin/codeset/Remote-Sensing-Image-Classification-master/checkpoints/CFEANet-model-bifpn-noattention99.52.pth')
    # model = torch.load(os.path.join('./checkpoints', config.checkpoint))
    # device = torch.device("cpu")
    model = torch.load(os.path.join('./', config.checkpoint), map_location=torch.device('cuda:0'))
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  , map_location=torch.device('cpu')
    print(next(model.parameters()).device)
    # map_location={'cuda:0':'cuda:2'}) , map_location=lambda storage, loc : storage.cuda(2) torch.device('cuda:2'), map_location=torch.device('cuda:2')
    # "/data3/qiaoxin/codeset/CFEANet-Submitted-to-IEEE-JSTARS-main_qx1/model-weight-ucm82/"
    # map_location={'cuda:1':'cuda:3'} , map_location='cuda:1'  , map_location='cuda:0'
    # print(model)
    # print(type(model))
    # model = torch.nn.DataParallel(model)

    # model.to(device)
    model.cuda()
    # model.eval()
    # model.cuda().eval()
    # print(next(model.parameters()).device)
    # model.eval()
    
    # validation data
    transform = transforms.Compose([transforms.Resize((config.width, config.height)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                         std=[0.229, 0.224, 0.225])])

    dst_valid = RSDataset('./huitu/RSSCN55/valid.txt', width=config.width,
                          height=config.height, transform=transform)
    dataloader_valid = DataLoader(dst_valid, shuffle=False, batch_size=int(config.batch_size/2), num_workers=config.num_workers)

    sum = 0
    val_top1_sum = 0
    labels = []
    preds = []
    model.eval()
    for ims, label in dataloader_valid:
        labels += label.numpy().tolist()

        input = Variable(ims).cuda()
        target = Variable(label).cuda()
        output = model(input)
        _, pred = output.topk(1, 1, True, True)
        preds += pred.t().cpu().numpy().tolist()[0]

        top1_val = accuracy(output.data, target.data, topk=(1,))
        
        sum += 1
        val_top1_sum += top1_val[0]
    avg_top1 = val_top1_sum / sum
    print('acc: {}'.format(avg_top1.data))

    labels_ = ['aGrass', 'bField', 'cIndustry', 'dRiverLake', 'eForest', 'fResident', 'fgParking']

    '''
    #labels-AID-30 = ['airport','bareland','baseballfield','beach','bridge', 'center','church','commercial','denseresidential','desert', 
                      'farmland','forest','industrial','meadow','mediumresidential','mountain','park','parking','playground','pond',               
                      'port','railwaystation','resort','river','school','sparseresidential','square','stadium','storagetanks','viaduct']             
               
    labels-WHU-19 = ['Airport', 'Beach', 'Bridge', 'Commercial', 'Desert', 'Farmland', 'footballField', 'Forest', 'Industrial', 'Meadow',
                     'Mountain', 'Park', 'Parking', 'Pond', 'Port', 'railwayStation', 'Residential', 'River', 'Viaduct']                             
              
    labels-WHU-19  = ['airport','beach','bridge', 'commercial', 'desert','farmland', 'footballfield','forest','industrial','meadow',  
            
    labels_RSSCN7 = ['aGrass', 'bField', 'cIndustry', 'dRiverLake', 'eForest', 'fResident', 'fgParking']
    
    labels_WHU-12 = ['agriculture', 'commercial', 'harbor', 'idle_land', 'industrial', 'meadow', 'overpass', 'park', 'pond', 'residential', 
                      'river', 'water']                                                                                                             

    labels-UCM-21 = ['agricultural','airplane', 'baseballdiamond','beach','buildings','chaparral', 'denseresidential', 'forest','freeway','golfcourse', 
                     'harbor','intersection', 'mediumresidential', 'mobilehomepark','overpass', 'parkinglot','river','runway', 'sparseresidential', 'storagetanks',  
                     'tenniscourt']
   
    labels-NWPU-45 = ['airplane', 'airport', 'baseball_diamond', 'caskerball_court', 'beach','bridge', 'chaparral', 'church', 'circular_farmland', 'cloud',  
                      'commercial_area', 'dense_residential', 'desert', 'forest', 'freeway','golf_course', 'ground_track_field', 'harbor', 'industrial_area', 'intersection',  
                      'island', 'lake', 'meadow', 'medium_residential', 'mobile_home_park', 'mountain', 'overpass', 'palace', 'parking_lot', 'railway',   
                      'storage_tank', 'tennis_court', 'terrace', 'thermal_power_station', 'wetland']                                                   
     
    labels-UCM-21 = ['Agricultural', 'Airplane', 'BaseballDiamond', 'Beach', 'Buildings', 'Chaparral', 'DenseResidential','Forest', 'Freeway', 'GolfCourse',
                     'Harbor', 'Intersection', 'MediumResidential', 'MobileHomePark','Overpass', 'ParkingLot', 'River', 'Runway', 'SparseResidential', 'StorageTanks',
                     'TennisCourt']                                                                                                                 
    '''
    plot_confusion_matrix(labels, preds, labels_)


if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    inference()