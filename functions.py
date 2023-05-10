import numpy as np
import cv2
from ICP import *
import matplotlib.pyplot as plt



def rotationMatrixToEulerAngles(R) :
 
    #assert(isRotationMatrix(R))
    sy = np.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
     
    singular = sy < 1e-6
 
    if  not singular :
        x = np.arctan2(R[2,1] , R[2,2])
        y = np.arctan2(-R[2,0], sy)
        z = np.arctan2(R[1,0], R[0,0])
    else :
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0
 
    return np.array([x, y, z])
    
def eulerAnglesToRotationMatrix(theta) :
     
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         np.cos(theta[0]), -np.sin(theta[0]) ],
                    [0,         np.sin(theta[0]), np.cos(theta[0])  ]
                    ])
         
         
                     
    R_y = np.array([[np.cos(theta[1]),    0,      np.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-np.sin(theta[1]),   0,      np.cos(theta[1])  ]
                    ])
                 
    R_z = np.array([[np.cos(theta[2]),    -np.sin(theta[2]),    0],
                    [np.sin(theta[2]),    np.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
                     
                     
    R = np.dot(R_z, np.dot( R_y, R_x ))
 
    return R
    
def loadFile(filename):
    f = open(filename,'r')
    L = f.readlines()
    f.close()
    R = []
    for l in L:
      R.append(l.split(";"))
    
    return R



def trajectoriesDrawer(pos, posReal,traj,mode,rfx0,rfy0):
    
   draw_x, draw_y = int((pos[0]-rfx0)*100000+200) , int((pos[1]-rfy0)*100000+200)
   real_x, real_y = int((posReal[0])-rfx0)*1000000+200,int((posReal[1]-rfy0)*1000000+200)
   print(posReal,pos)
   #If street_view
   # ~ draw_x, draw_y = int(pos[0]*5)+300, int(pos[1]*5)+70
   # ~ real_x, real_y = int(posReal[0]*5)+300, int(posReal[1]*5)+70
   if not(mode):
        cv2.circle(traj, (draw_x,draw_y), 1, (0,0,255), 1)
        cv2.putText(traj,"LEARNING MODE",(50,100),cv2.FONT_HERSHEY_SIMPLEX,.5,(0, 0, 0, 255),1)
        cv2.putText(traj,"CORRECTION MODE",(50,100),cv2.FONT_HERSHEY_SIMPLEX,.5,(209, 80, 0, 255),1)
   else:
        cv2.putText(traj,"LEARNING MODE",(50,100),cv2.FONT_HERSHEY_SIMPLEX,.5,(209, 80, 0, 255),1)
    
   cv2.circle(traj, (real_x,real_y), 1 , (0,255,0), 1)
   cv2.rectangle(traj, (10, 20), (600, 60), (0,0,0), -1)
   
   return traj

def trajectoriesDrawerPlt(pos,posReal):
    print(posReal)
    #plt.plot(pos[:,0],pos[:,1],'r')
    plt.plot(posReal[:,0],posReal[:,1],'g')
    plt.draw()
    plt.pause(0.01)
         

def calibrate(pos,old_pos,posReal,old_posReal):
         print("!______BEGIN_CALIBRATION____!")
         vo_vect = pos - old_pos
         real_vect = posReal - old_posReal
         real_head = np.arctan2(real_vect[1],real_vect[0])
         deltaAngle = real_head - np.arctan2(vo_vect[1],vo_vect[0])
         
         rotationMatrix = eulerAnglesToRotationMatrix([0,0,deltaAngle])
         pos = np.dot(rotationMatrix,pos)
         
         translationVector = posReal  - pos
         pos = pos +translationVector
         
         print("!______ENDED_CALIBRATION____!")
         
         return pos, rotationMatrix, translationVector
         
def calibrateV2(posReal,old_posReal,vo):
         print("!______BEGIN_CALIBRATION____!")

         odometryAngles = rotationMatrixToEulerAngles(vo.cur_R)
         real_vect = posReal - old_posReal
         real_head = np.arctan2(real_vect[1],real_vect[0])
         deltaAngle = real_head - odometryAngles[1]
        
            #real_head+np.pi+1 for images1 set   
            #real_head+np.pi-0.22 for images2 set
            #real_head+np.pi+np.pi/2 for images3 set
            #real_head+np.pi+np.pi/2 for images4 set
         odometryAngles = np.array([odometryAngles[0],real_head+np.pi+1,odometryAngles[2]])
         rotationMatrix = eulerAnglesToRotationMatrix(odometryAngles)
        
         vo.cur_R = rotationMatrix
         vo.cur_t[0] = np.array([[posReal[0]]])
         vo.cur_t[2] = np.array([[posReal[1]]])

         print("!______ENDED_CALIBRATION____!")
         return vo

    
def calibrateV23(refPos,vo):
         print("!______CORRECTION____!")

         odometryAngles = rotationMatrixToEulerAngles(vo.cur_R)

         #odometryAngles = np.array([odometryAngles[0],head+np.pi/2,odometryAngles[2]])
         rotationMatrix = eulerAnglesToRotationMatrix(odometryAngles)
         
         #vo.cur_R = rotationMatrix
         dX = refPos[0] - vo.cur_t[0]
         dY = refPos[1] - vo.cur_t[2]
         vo.cur_t[0] = np.array([[refPos[0]]])
         vo.cur_t[2] = np.array([[refPos[1]]])         #vo.cur_R = rotationMatrix

         print("!______CORRECTION____!")

         return vo
   
