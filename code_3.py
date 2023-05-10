"""
Created on Wed Nov 6 22:35:49 2019

@author: dali
"""
import time


from inspireV4 import *
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

odoPositions = np.array([0,0,0])
pos = np.array([0,0,0])
cur_pos = None

with open('intrisic_chiha.npy', 'rb') as f:
    intrinsic = np.load(f)
    
gt_path = []
estimated_path = []
camera_pose_list = []
start_pose = np.ones((3,4))
start_translation = np.zeros((3,1))
start_rotation = np.identity(3)
start_pose = np.concatenate((start_rotation, start_translation), axis=1)

skip_frames = 2
DATASET_ROOT = "./image2/"

vo = CameraPoses(DATASET_ROOT, skip_frames, intrinsic)
#cap = cv2.VideoCapture('http://192.168.43.1:8080/video')

process_frames = False
old_frame = None
new_frame = None
index = 0

cur_pose = start_pose


while (index<130) :
    
    print(index)
    #img = cv2.imread(DATASET_ROOT+str(index)+'.png',cv2.IMREAD_UNCHANGED)
    new_frame = cv2.imread(DATASET_ROOT+str(index)+'.png',cv2.IMREAD_UNCHANGED)

    if process_frames :
        q1, q2 = vo.get_matches(old_frame, new_frame)
        if q1 is not None:
            if len(q1) > 20 and len(q2) > 20:
                transf = vo.get_pose(q1, q2)
                cur_pose = cur_pose @ transf
                pos = np.array([cur_pose[0, 3], cur_pose[2, 3]])
        
        hom_array = np.array([[0,0,0,1]])
        hom_camera_pose = np.concatenate((cur_pose,hom_array), axis=0)
        camera_pose_list.append(hom_camera_pose)
        estimated_path.append((cur_pose[0, 3], cur_pose[2, 3]))
        
        
        estimated_camera_pose_x, estimated_camera_pose_y = cur_pose[0, 3], cur_pose[2, 3]


    
    old_frame = new_frame
    process_frames = True
    index = index + 1   
    

    cv2.imshow("img", new_frame)

take_every_th_camera_pose = 1

estimated_path = np.array(estimated_path[::take_every_th_camera_pose])
plt.plot(estimated_path[:,0],estimated_path[:,1])
plt.show()
