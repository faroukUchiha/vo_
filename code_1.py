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

def learningModeDisabler():
    global referenceImage, img_pil, LEARNING_MODE,vo,refIndex,CNN_FREED, lastCorrectionIndex,lastEstimationIndex 
    CNN_FREED = False
    output1,output2 = network(img_pil,referenceImage)
    result = F.pairwise_distance(output1, output2)
    CNN_FREED = True
    print("CNN RESULT", result)
    
    if result<0.3 :
      LEARNING_MODE = False
      print("__LEARNING__MODE__DISABLED__")
      lastCorrectionIndex = 1
      lastEstimationIndex = 1
      refIndex = 1
      
      #VO INIT
      camSpecs = PinholeCamera(720,1280, 1.16537560e+03, 1.15127733e+03,3.18015075e+02, 6.50365937e+02)
      vo = VisualOdometry(camSpecs)
      
      (threading.Thread(target = loopClosureDetector)).start()
  
def storingImages():
    global groundTruthDictionary,img,posReal,old_posReal,realHead, index,learningPositionList

    refImg = transformCompose(I.fromarray(img))
    refImg = refImg.unsqueeze(0)

    #print("*******SOUTENANCE_IMG_SIZE******",sys.getsizeof(refImg.storage()))
    #print("*******SOUTENANCE_POS******",sys.getsizeof(tuple(learningPositionList[-1])))

    #vec_real = posReal - old_posReal
    #head_real = np.arctan2(vec_real[1],vec_real[0])
    groundTruthDictionary[tuple(learningPositionList[-1])] = refImg

def loopClosureDetector():
   
   global groundTruthDictionary, pos, old_pos, translationVector ,vo ,refIndex,lost_counter ,posReal ,odoPositions ,realPositions,lastCorrectionIndex,learningPositionList,old_minimum_index, old_T,withoutCorrection
  
   #Stored reference positions
   points = np.asarray(list(groundTruthDictionary.keys()))
   #current estimated position
   node = np.array(pos)
   #measure distances between current estimated position and all ref positions
   dist_2 = np.sum((np.array(learningPositionList)[:,:3] - node)**2, axis=1)
   #Index of the closest distance
   minimum_index = np.argmin(dist_2)
   print("MIN INDX = ",minimum_index)
   #minimum_distance
   minimum_distance = dist_2[minimum_index]

   lost_counter = lost_counter + 1
   
   with torch.no_grad(): #sans gradient
      if minimum_distance < 8:
         
         referencePosition = learningPositionList[minimum_index] 
         old_refPos = learningPositionList[minimum_index-1]
         lookForKey =  np.sum((points[:,:3] - referencePosition)**2, axis=1)
         closestKeyIndex = np.argmin(lookForKey)
         referenceKey = tuple(points[closestKeyIndex])
         referenceImage = groundTruthDictionary[referenceKey]
         #minimum_index = learningPositionList.index(list(referencePosition))
         #Estimate deviation
         translationVector_correction = np.array(referencePosition[:3])- node
         
         CNN_FREED = False
         output1,output2 = network(img_pil,referenceImage)
         similarity = F.pairwise_distance(output1, output2)
         CNN_FREED = True
         
         OFFSET = 0

         #if under two images are similar so there is loop closure
         if similarity <= 0.3: #0.37
         
                    if old_minimum_index == -1000:
                        old_minimum_index = OFFSET
                    
                    print("LOOP CLOSURE")
                    closedLoop = True
                    
                    vo  = calibrateV23(referencePosition[:3],vo)

                    lost_counter = 0

                    subEstimate = odoPositions[lastCorrectionIndex:len(odoPositions)]
                    
                    M = old_minimum_index#lastCorrectionIndex%len(learningPositionList) + OFFSET
                    N = len(subEstimate)#%len(learningPositionList)
                    
                    if (M + N)<len(learningPositionList):
                        subReal = learningPositionList[M:M+N]
                    else:
                        upper = learningPositionList[M:]
                        X = N - len(upper)
                        subReal = upper + learningPositionList[:X]

                    subEstimate, subReal = np.array(subEstimate), np.array(subReal)

                    print("EstSHP= ",subEstimate.shape,"--- RealSHP= ",subReal.shape)


                    T,_,_ = icp(subEstimate,subReal)
                    subEstimate  = np.dot(T[0:3,0:3],subEstimate.T)
                    subEstimate  = T[0:3,3] + subEstimate.T
                    
                    for i in range(lastCorrectionIndex,len(odoPositions)):
                        (odoPositions[i])  = (subEstimate[i-lastCorrectionIndex])
                        
                  
                    print(odoPositions[i])
                    odoPositions[lastCorrectionIndex] = 0.5*(odoPositions[lastCorrectionIndex+1] + odoPositions[lastCorrectionIndex-1])
                    lastCorrectionIndex = len(odoPositions)
                    old_minimum_index = minimum_index
                    old_T = T

#GLOBAL VARS
DATASET_ROOT = "./images/"
DATA_FILENAME = DATASET_ROOT + "values.txt"

RUNNING_PROCESS = True
RESTARTED = False
#586 for images1 set / #1014 for images2 set / 370 for images3 set
RESTARTING_INDEX = 50
LEARNING_MODE = True

CNN_FREED = True


##YOLO 
PATH = "./network_YOLO"
network = SiameseYoloV()


network.load_state_dict(torch.load(PATH,map_location='cpu'))
network.eval()

traj = np.zeros((600,600,3), dtype=np.uint8)

#USEFUL VARS
index = 1
refIndex = 0
data = loadFile(DATA_FILENAME)
t0 = float(data[0][4])
pos = np.array([0,0,0])
posReal = np.array([float(data[0][1]),float(data[0][2]),float(data[0][3])])
real_head = 0

travelDistance = 0
lost_counter = 0
lastCorrectionIndex = 0
lastEstimationIndex = 0
old_minimum_index = -1000
#Deep Learning useful vars
groundTruthDictionary =  collections.OrderedDict()
transformCompose = transforms.Compose([transforms.Resize(size=(224,224)), transforms.ToTensor()])
learningPositionList = []

##ARRAYS TO STORE POSITIONS AND ERRORS
odoPositions = []
realPositions = []
withoutCorrection = []
old_T = np.eye(4)

   
GTPL_size = np.array([])
RID_size =np.array([])

LSR = .125
CSR = .5




while(RUNNING_PROCESS):
   print(index)
   ##GETTING IMAGE
   img = cv2.imread(DATASET_ROOT+str(index)+'.png',cv2.IMREAD_UNCHANGED)
   img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
   
   img_pil = transformCompose(I.fromarray(img))
   img_pil = img_pil.unsqueeze(0)
   
   ##GETTING DATA
   current_data = data[index]
   if index==1:
      rfx0,rfy0=float(current_data[1]), float(current_data[2])
   
   timestamp = float(current_data[4])
   #altitude = float(current_data[2])
   realX = float(current_data[1])
   realY = float(current_data[2])
   realZ = float(current_data[3])
   #speed = float(current_data[6])
   speed = 0.1
   dT = timestamp - t0
   travelDistance = travelDistance + speed*dT
   
   #realX=(round(realX,3)-realX)*1000000
   #realY=(realY-round(realY,3))*100000
   posReal = np.array([realX,realY,realZ])
   print(posReal,pos)

   if index == 1 and not(RESTARTED):
        referenceImage = transformCompose(I.fromarray(img))
        referenceImage = referenceImage.unsqueeze(0)

   if index>1:
      RESTARTED = False

   if LEARNING_MODE:
            learningPositionList.append(list(posReal))
            #GTPL_size.append(sys.getsizeof(learningPositionList))
            size = np.array([sys.getsizeof(learningPositionList)])
            GTPL_size = np.append(GTPL_size, size)
            #RID_size.append(sys.getsizeof(groundTruthDictionary))
            size1 = np.array([sys.getsizeof(groundTruthDictionary)]) 
            RID_size = np.append(RID_size, size1)
           #start=time.time()

            if travelDistance>LSR: #0.5
               print("__LEARNING__")
               travelDistance = 0
               storingImages()
               if index>99999 and CNN_FREED:
                  (threading.Thread(target = learningModeDisabler)).start()
   else:
            #ESTIMATING POSITION FROM VISUAL ODOMETRY
            vo.update(img_gray,index,speed*dT)
            if(refIndex ==1):
               print(GTPL_size[-1]/1024)
               print(RID_size[-1]/1024)
               print('leng_GTPL = ',len(learningPositionList))
               print('leng_RID = ', len(groundTruthDictionary))
               penteGTPL = ((GTPL_size[-1])-GTPL_size[0])/1024 / (len(GTPL_size)-1)  
               penteRID = ((RID_size[-1])-RID_size[0])/1024 / (len(RID_size)-1)  
               plt.figure()
               GTPL_size = np.array(GTPL_size)/(1024)
               RID_size = np.array(RID_size)/(1024)
               Xaxis = np.arange(0,len(GTPL_size))
               
               approxGTPL = (penteGTPL*Xaxis) 
               approxGTPL[:]= approxGTPL[:] + GTPL_size[0]
               
               approxRID = (penteRID*Xaxis) 
               approxRID[:] = approxRID[:] + RID_size[0]
               
               plt.xlabel('Iterations')
               plt.ylabel('Size in KB')
               plt.plot(Xaxis,GTPL_size, color="red", label="GTPL size")
               plt.plot(Xaxis,RID_size, color="green", label="RID size")
               plt.plot(Xaxis,approxGTPL, '--r', label="GTPL size approximation")
               plt.plot(Xaxis,approxRID, '--g', label="RID size approximation")
               plt.legend()
               plt.show() 
               sleep(.25)
            #Visual odometry Position
            if refIndex > 1:
               cur_pos = vo.cur_t
               while(cur_pos is None):
                  vo.update(img_gray,index,speed*dT)
                  cur_pos = vo.cur_t
               
               x , y , z = cur_pos[0][0], cur_pos[1][0], cur_pos[2][0]
               #x , y , z = cur_pos[0][0], 1, cur_pos[2][0]

               pos = np.array([x,z,y])
               
               if(refIndex==5):
                     print("ALIGN")
                     vo = calibrateV2(posReal,old_posReal,vo)
   
               #Start processing
               if(refIndex>7):
                     start=time.time()
                     RESTARTED = False                     
                  
                     ##LoopClosure Detection Threadx
                     if travelDistance>CSR and CNN_FREED: #travel>2
                        travelDistance = 0
                        print("__DETECTING__")
                        #(threading.Thread(target = loopClosureDetector)).start()
                        #loopClosureDetector()
                        print("lost counter ", lost_counter)
                     
                     if lost_counter>5:
                        #LEARNING_MODE  = True
                        lost_counter = 0
                     
                     odoPositions.append(pos)
                     realPositions.append(posReal)
                     withoutCorrection.append(pos)
                     end=time.time()
                     print("process time=",end-start)


                     
                     if (index==RESTARTING_INDEX-2):
                         #end=time.time()
                         #print("process time=",end-start)
                         plt.close('all')
                         A = np.array(odoPositions[1:])
                         B = np.array(realPositions[1:])
                         C = np.array(withoutCorrection[1:])
                        
                         #2D TRAJ PLOT
                         plt.figure()
                         plt.xlabel(' X (m) ')
                         plt.ylabel(' Y (m) ')
                         plt.plot(A[:,0],A[:,1], color = "red", label="ICP")
                         plt.plot(B[:,0],B[:,1], color = "green", label="Real Data")
                         plt.plot(C[:,0],C[:,1], color = "blue", label="without ICP")
                         plt.legend()                        
                        
                         #3D TRAJ PLOT
                         fig=plt.figure()
                         #ax = plt.gca(projection='3d')
                         ax = fig.add_subplot(111, projection='3d')

                         ax.set_xlabel(' X (m) ')
                         ax.set_ylabel(' Y (m) ')
                         ax.set_zlabel(' Z (m) ')
                         plt.plot(A[:,0],A[:,1],A[:,2], color = "red", label="ICP")
                         plt.plot(B[:,0],B[:,1],B[:,2], color = "green", label="Real Data")
                         plt.plot(C[:,0],C[:,1],C[:,2], color = "blue", label="without ICP")
                         plt.legend()
                        
                        
                         #DRIFT PLOT
                         plt.figure()
                         T = np.arange(0,len(A[:,0]))
                         dX = (B[:,0] - A[:,0])
                         dY = (B[:,1] - A[:,1])
                         dZ = (B[:,2] - A[:,2])
                         dL = np.sqrt(dX**2+dY**2+dZ**2)
                         dLM= sum(dL)/len(dL)
                         dLM = [dLM]*len(dL)
                         #plt.plot(T,dX, color = "red", label="delta X")
                         #plt.plot(T,dY, color = "green", label="delta Y")
                         #plt.plot(T,dZ, color = "orange", label="delta Z")
                         plt.xlabel(' Number of iterations ')
                         plt.ylabel(' Drift (m) ')
                         plt.plot(T,dL, color = "red", label="Drift using ICP")
                         plt.plot(T,dLM, '--r', label="Average drift using ICP")
                         plt.legend()

                         #np.savez("paper_resn_0", dL, dLM)

                         print("Average drift using ICP",dLM)
                        
                         dX = (B[:,0] - C[:,0])
                         dY = (B[:,1] - C[:,1])
                         dZ = (B[:,2] - C[:,2])
                         dL = np.sqrt(dX**2+dY**2+dZ**2)
                         dLM= sum(dL)/len(dL)
                         dLM = [dLM]*len(dL)
                         plt.plot(T,dL, color = "blue", label="Drift without ICP")
                         plt.plot(T,dLM, '--b', label="Average drift without ICP")
                         plt.legend()
                         print("Average drift without ICP",dLM)
                         plt.show()

                     
                         plt.pause(0.0001)
                     
                  
            refIndex = refIndex + 5
               ##POI DRAWER
            points = vo.px_ref
            for j in range(0,len(points)):
               cv2.circle(img,(int(points[j][0]),int(points[j][1])),2,(0,255,255),1)

  


      

         

  
         #Draw trajectories using OpenCV
   traj = trajectoriesDrawer(pos,posReal,traj,LEARNING_MODE,rfx0,rfy0)
   #realPositions.append(posReal)
   #trajectoriesDrawerPlt(np.array(odoPositions[1:]),np.array(realPositions[1:]))       





         
         

   
   #Saving old values
   old_pos, old_posReal = pos, posReal
   old_realHead = real_head
   #t0 = timestamp

   
   
   #Show opencv trajectories and frames
   cv2.imshow('Trajectory', traj)
   cv2.moveWindow('Trajectory',700,0)
   cv2.imshow('frame',img)
   cv2.moveWindow('frame',200,0)
   
   index = index +1
   
   if(index==RESTARTING_INDEX):
      index = 1
      t0 = float(data[0][4])
      RESTARTED = True
      LEARNING_MODE = False
      
      if(True): #DATASET_ROOT == "./images2/"):
         #VO INIT
         camSpecs = PinholeCamera(720,1280, 1.16537560e+03, 1.15127733e+03,3.18015075e+02, 6.50365937e+02)
         vo = VisualOdometry(camSpecs)
         (threading.Thread(target = loopClosureDetector)).start()
         #traj = np.zeros((600,600,3), dtype=np.uint8)
      
   
   if cv2.waitKey(33) == 120:
      RUNNING_PROCESS = False


