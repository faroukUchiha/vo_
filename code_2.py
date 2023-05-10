"""
Created on Wed Nov 6 22:35:49 2019

@author: dali
"""
import time


from inspireV3 import *
from functions import *
from ICP import *


import collections
import numpy as np
from PIL import Image as I
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

from Siamese import *
from SiameseBeta import *

import threading

import sys

from time import sleep
fig, ax = plt.subplots(figsize=(10, 10))
def plot(pt,color="#FF0000",marker="x"):
    #ax.scatter(pt[0],pt[1],marker=marker,color=color)
    ax.plot(pt[:,0],pt[:,1])
    plt.draw()
    plt.pause(0.9)



transformCompose = transforms.Compose([transforms.Resize(size=(224,224)), transforms.ToTensor()])

traj = np.zeros((600,600,3), dtype=np.uint8)
cv2.rectangle(traj, (10, 20), (600, 60), (0,0,0), -1)

odoPositions = np.array([0,0,0])
pos = np.array([0,0,0])
cur_pos = None
DATASET_ROOT = "./image2/"
camSpecs = PinholeCamera(720,1280, 1.16537560e+03, 1.15127733e+03,3.18015075e+02, 6.50365937e+02)
vo = VisualOdometry(camSpecs)
index = 0
while (index<130):
    
    print(index)
    #img = cv2.imread(DATASET_ROOT+str(index)+'.png',cv2.IMREAD_UNCHANGED)
    img = cv2.imread(DATASET_ROOT+str(index)+'.png',cv2.IMREAD_UNCHANGED)


    img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
   
    img_pil = transformCompose(I.fromarray(img))
    img_pil = img_pil.unsqueeze(0)
    
    
    if index > 1:   
        cur_pos = vo.cur_t
        while(cur_pos is None):
            vo.update(img_gray,index,1)
            cur_pos = vo.cur_t
        vo.update(img_gray,index,.25)
        
        x ,z , y = cur_pos[0][0], cur_pos[1][0], cur_pos[2][0]
    
        pos = np.array([x,y,z])
        odoPositions= np.vstack((odoPositions,pos))
    
        print(pos)
        plot(odoPositions)
        
        traj =  cv2.circle(traj, (int(pos[0])*5+300, int(pos[1])*5+300), 1, (0,0,255), 1)
        points = vo.px_ref
        #for j in range(0,len(points)):
        #   cv2.circle(img,(int(points[j][0]),int(points[j][1])),2,(0,255,255),1)
        cv2.imshow('frame',img)
        #cv2.imshow("traj",traj)
        #cv2.moveWindow('frame',200,0)

    index = index + 1   

