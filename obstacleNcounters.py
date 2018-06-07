

import numpy as np
import time
import cv2
from pathInertiaModel import pathInertiaModel
from socket import socket, gethostbyname, AF_INET, SOCK_DGRAM
from sklearn.neighbors import KNeighborsClassifier
import copy
from random import randint
import skimage.measure
import keras
import math
import threading

def obstacleNcounters():
    def loadModel():
        loaded_model = None
        with open("pathInertia.json", "r") as json_file:
            loaded_model_json = json_file.read()
            loaded_model = keras.models.model_from_json(loaded_model_json)
            loaded_model.load_weights("pathInertia.h5")
            print("Model loaded from disk")
        return loaded_model
    
    def calcParameters(ocp):
        setpara = []
        pos1 = ocp[0]
        for i in range(1,10):
            pos2 =  ocp[1]
            y = pos2[1] - pos1[1]
            x = pos2[0] - pos1[0]
            disp = round(y**2 + x**2,1)
            slp = round(math.atan2(y,x),2)
            dr=1
            if slp<0:
                dr=0
            setpara.append([disp, slp, dr])
            pos1 = pos2
        return setpara
    
    def pointV(orgP, predP):
        y = orgP[1]*(math.sin(predP[1]))
        x = orgP[0]*(math.cos(predP[1]))
        if not predP[2]:
            y=-1*y
        return [x, y]
    
    def proc1():
#        label1 = []
#        out1 = []
#        obstPred1 = []
        mainout = skimage.measure.block_reduce(scr, (5,5,1), np.mean)
        for h in range(0,50,2):
            for w in range(0,136,2):
                info = [mainout[h,w, 0] , mainout[h,w+1,0]  , mainout[h+1,w,0] ,mainout[h+1,w+1,0] ,
                           mainout[h,w, 1] , mainout[h,w+1,1]  , mainout[h+1,w,1] ,mainout[h+1,w+1,1] ,
                           mainout[h,w, 2] , mainout[h,w+1,2]  , mainout[h+1,w,2] ,mainout[h+1,w+1,2]]
                a=(h+1)*5
                b=(w+1)*5
                label1.append([a*1000+b])
                out1.append(info)
                if not beg:
                    ind_pred = classifier.predict(np.reshape(info,(-1,12)))
                    ind_pred = int(ind_pred)
                    if  len(prev_info_countours[ind_pred]) >= 10:
                        sendInertia = True
                        del prev_info_countours[ind_pred][0]
                    info_countours[a*1000+b] = prev_info_countours[ind_pred] + [[a,b]]
#                    pointsParameter = calcParameters(info_countours[a*1000+b])
#                    pred = loaded_model.predict(np.reshape(pointsParameter,(-1,9,3)))
#                    obstPred1.append([[a,b],pointV([a,b],pred[0])])
                else:
#                    print(a*1000+b)
#                    print(info_countours[a*1000+b])
                    info_countours[a*1000+b] = [[a,b]]
    
    def proc2():
#        global label2 
#        global out2 
#        global obstPred2 
        mainout1 = skimage.measure.block_reduce(scr[2:,2:], (5,5,1), np.mean)
        for h in range(0,49,2):
            for w in range(0,135,2):
                info = [mainout1[h,w, 0] , mainout1[h,w+1,0]  , mainout1[h+1,w,0] ,mainout1[h+1,w+1,0] ,
                           mainout1[h,w, 1] , mainout1[h,w+1,1]  , mainout1[h+1,w,1] ,mainout1[h+1,w+1,1] ,
                           mainout1[h,w, 2] , mainout1[h,w+1,2]  , mainout1[h+1,w,2] ,mainout1[h+1,w+1,2]]
                a = h*5+7
                b = w*5+7
                label2.append([a*1000+b])
                out2.append(info)
                if not beg:
                    ind_pred = classifier.predict(np.reshape(info,(-1,12)))
                    ind_pred = int(ind_pred)
                    if  len(prev_info_countours[ind_pred]) >= 10:
                        sendInertia = True
                        del prev_info_countours[ind_pred][0]
                    info_countours[a*1000+b] = prev_info_countours[ind_pred] + [[a,b]]
                    
                else:
                   info_countours[a*1000+b] = [[a,b]]

           
    PORT_NUMBER = 4000
    SIZE = 1024*1024
#    hostName = gethostbyname( '0.0.0.0' )
#    mySocket = socket( AF_INET, SOCK_DGRAM )
#    mySocket.bind( (hostName, PORT_NUMBER) )
    print ("Test server listening on port {0}\n".format(PORT_NUMBER))
    
    classifier = KNeighborsClassifier(n_neighbors=1)  
#    kernel = np.ones((3,3),np.uint8)
    print("start")
    prev_info_countours = []
    info_countours = list(np.zeros((261680,10,2)))
    beg = 1
    vrec = cv2.VideoCapture('v4.mp4',0)
    sendInertia = False
    k = 300*6*8
    obstPred = []
    loaded_model = loadModel()
    
    while k:
        ret, scr = vrec.read()
        k-=1
    print("running")
    while True:
        label1 = []
        out1 = []
        obstPred1 = []   
        label2 = []
        out2 = []
        obstPred2 = [] 
        label = []
        out = []
        obstPred = []
#        cv2.destroyAllWindows()
        start_time = time.time()
        ret, scr = vrec.read()
        ret, scr = vrec.read()
        if not ret:
            break
#        scr = None
#        while not scr:
#            (scr,addr) = mySocket.recvfrom(SIZE)
    
        scr = cv2.resize(scr, (680,480)) 
        scr = np.array(scr[220:480,:])
#        cv2.imshow('scr1',scr)
        
        blank = cv2.imread('blank.png', 0)
        blank = cv2.resize(blank,(680,260))
    
        frame = cv2.GaussianBlur(scr, (5, 5), 0)
        scr = cv2.addWeighted(scr,2.5,frame,-1.5,0)
#        scr = cv2.bilateralFilter(scr,9,75,75)
        
#        cv2.imshow('scred',scr)
        
#        
#        label = []
#        out = []
#        obstPred = []
#        mainout = skimage.measure.block_reduce(scr, (5,5,1), np.mean)
##        mainout1 = skimage.measure.block_reduce(scr[2:,2:], (5,5,1), np.mean)
#        for h in range(0,50,2):
#            for w in range(0,136,2):
#                info = [mainout[h,w, 0] , mainout[h,w+1,0]  , mainout[h+1,w,0] ,mainout[h+1,w+1,0] ,
#                           mainout[h,w, 1] , mainout[h,w+1,1]  , mainout[h+1,w,1] ,mainout[h+1,w+1,1] ,
#                           mainout[h,w, 2] , mainout[h,w+1,2]  , mainout[h+1,w,2] ,mainout[h+1,w+1,2]]
#                a=(h+1)*5
#                b=(w+1)*5
#                label.append([a*1000+b])
#                out.append(info)
#                if not beg:
#                    ind_pred = classifier.predict(np.reshape(info,(-1,12)))
#                    ind_pred = int(ind_pred)
#                    if  len(prev_info_countours[ind_pred]) >= 10:
#                        sendInertia = True
#                        del prev_info_countours[ind_pred][0]
#                    info_countours[a*1000+b] = prev_info_countours[ind_pred] + [[a,b]]
#                    pointsParameter = calcParameters(info_countours[a*1000+b])
#                    pred = loaded_model.predict(np.reshape(pointsParameter,(-1,9,3)))
#                    obstPred.append([[a,b],pointV([a,b],pred[0])])
#                else:
##                    print(a*1000+b)
##                    print(info_countours[a*1000+b])
#                    info_countours[a*1000+b] = [[a,b]]
#         
#            
#            
#        for h in range(0,49,2):
#            for w in range(0,135,2):
#                info = [mainout1[h,w, 0] , mainout1[h,w+1,0]  , mainout1[h+1,w,0] ,mainout1[h+1,w+1,0] ,
#                           mainout1[h,w, 1] , mainout1[h,w+1,1]  , mainout1[h+1,w,1] ,mainout1[h+1,w+1,1] ,
#                           mainout1[h,w, 2] , mainout1[h,w+1,2]  , mainout1[h+1,w,2] ,mainout1[h+1,w+1,2]]
#                a=(h*5+7)
#                b=w*5+7
#                label.append([a*1000+b])
#                out.append(info)
#                if not beg:
#                    ind_pred = classifier.predict(np.reshape(info,(-1,12)))
#                    ind_pred = int(ind_pred)
#                    if  len(prev_info_countours[ind_pred]) >= 10:
#                        sendInertia = True
#                        del prev_info_countours[ind_pred][0]
#                    info_countours[a*1000+b] = prev_info_countours[ind_pred] + [[a,b]]
#                else:
#                   info_countours[a*1000+b] = [[a,b]]
        
        th1 = threading.Thread(target = proc1, args=())
        th2 = threading.Thread(target = proc2, args=())
        th1.start()
        th2.start()
        th1.join()
        th2.join()
        out = list(out1) + list(out2)
        label = list(label1) + list(label2)
        obstPred = list(obstPred1) + list(obstPred2)
#                    
#        label = np.array(label).reshape(-1,1)         
#        out = np.array(out).reshape(-1,12)

#        if sendInertia:
##            obsPred = pathInertiaModel(info_countours)
#            for n in range(len(info_countours)):
#                blk = cv2.imread('blank.png',1)
#                blk = cv2.resize(blk,(680,260))
#                cl1 = randint(0, 230)
#                cl2 = randint(0, 230)
#                cl3 = randint(0, 230)
#                for pts in info_countours[n]:
#                    cv2.circle(blk,(int(pts[1]),int(pts[0])), 2,( cl1, cl2, cl3), -1)
#                cv2.circle(blk,(int(obstPred[n][1][1]),int(obstPred[n][1][0])), 2, (cl1,cl2,cl3), -1)
#                cv2.imshow('blk',blk)
#                time.sleep(0.02)
#        cv2.imshow('blk',blk)

        beg = 0
        classifier.fit(out,label)
        prev_info_countours = copy.deepcopy(info_countours)
        print("FPS: ", 1.0 / (time.time() - start_time))
    #     time.sleep(0.2)
    #     print("FPS: ", 1.0 / (time.time() - start_time))
        q=cv2.waitKey(1)
#        if q == 27:
#            cv2.destroyAllWindows()
#            break
    cv2.destroyAllWindows()
obstacleNcounters()