

import numpy as np
import time
import cv2
import threading
from pathInertiaModel import pathInertiaModel
from socket import socket, gethostbyname, AF_INET, SOCK_DGRAM
import sys

def obstacleNcounters():
    def hod_rd_sub1(mean):
        for i in range(my,shape[1]-lmd,lmd-2):
            ver_rd(i,mean)
    
    def hod_rd_sub2(mean1):
        for i in range(my+lmd,0, -(lmd-2)):
            ver_rd(i,mean1)
    
    
    def hor_rd(mean):
        hod_rd_sub1(mean)
        hod_rd_sub2(mean)
    
    
    def ver_rd(x,mean):
        global lim1
        global lim2
        global lmd
        lm1 = lim1
        lm2 = lim2
        for i in range(mx,lmd+1,-(lmd-10)):
            lm1 += lm1*0.03
            lm2 += lm2*0.03
            low = mean - [0,lm1,lm2]
            high = mean + [180,lm1,lm2]
            newscr[i-lmd:i, x:x+lmd] = cv2.inRange(hls[i-lmd:i, x:x+lmd],low,high)
            res = [0]+[np.mean(hls[i-lmd:i, x:x+lmd,j]) for j in range(1,3)]
            res = np.array(res)
            if(abs(res[1]-mean[1])<=15 and abs(res[2]-mean[2])<=15):
                mean = (4*res+6*mean)/10
    
    # lim = 4*std
    lmd = 40
    lim1 = lim2 = 20
    kernel = np.ones((3,3),np.uint8)
    
    shape = [480,680]#scr.shape
    scr2 = 0
    mx = shape[0]
    my = shape[1]//2
    print("running")
#    kkk = 300*8*6
    mask_prev1 = np.zeros((shape[0],shape[1]),np.uint8)
#    mask_prev2 = np.zeros((shape[0],shape[1]),np.uint8)
    mean_prev = 0
#    vrec = cv2.VideoCapture('v4.mp4',0)
    
#    while(kkk):
#        kkk-=1
#        ret, scr = vrec.read()
    PORT_NUMBER = 4000
    SIZE = 1024*1024
    hostName = gethostbyname( '0.0.0.0' )

    mySocket = socket( AF_INET, SOCK_DGRAM )
    mySocket.bind( (hostName, PORT_NUMBER) )

    print ("Test server listening on port {0}\n".format(PORT_NUMBER))
    
    while True:
#        start_time = time.time()
        
        obstacleContoursPts = []
        scr = None
        while not scr:
            (scr,addr) = mySocket.recvfrom(SIZE)
#        ret, scr = vrec.read()
#        if not ret:
#                break
        scr = cv2.resize(scr, (680,480)) 
        scr2 = scr
        newscr = np.zeros((shape[0],shape[1]),np.uint8)
    #     scr = cv2.bilateralFilter(scr,7,12,12)
        hls = cv2.cvtColor(scr, cv2.COLOR_BGR2HLS)
        mean = [0]+[np.mean(hls[440:480,200:480, i]) for i in range(1,3)]
        std = [np.std(hls[440:480,200:480, i]) for i in range(1,3)]
        lim1 = (lim1 + 16*std[0])/5  #(1:4(4))/5
        lim2 = (lim2 + 16*std[1])/5

        mean = np.array(mean)
        mean = 0.6*mean + mean_prev*0.4
        mean_prev = mean

        hor_rd(mean)
        
        gray = cv2.cvtColor(scr, cv2.COLOR_BGR2GRAY)
        mean1 = np.mean(gray[440:480,200:480])
        mean1 = np.array(mean1)
        low1 = mean1 - 25
        high1 = mean1 + 25
        
        mask1 = cv2.inRange(gray,low1,high1)

    #     newscr = cv2.morphologyEx(newscr,cv2.MORPH_OPEN,kernel, iterations = 1)
    #     newscr = cv2.morphologyEx(newscr,cv2.MORPH_CLOSE,kernel, iterations = 1)
        
            
        mask3 = newscr+mask1
    #     band3 =  cv2.bitwise_and(scr,scr,mask = mask3)
        
        mask4 = mask_prev1 + mask3 
    #     mask_pres2 = mask_prev1
        mask_prev1 = mask3 
            
        mask4 = cv2.morphologyEx(mask4,cv2.MORPH_OPEN,kernel, iterations = 1)
        mask4 = cv2.morphologyEx(mask4,cv2.MORPH_CLOSE,kernel, iterations = 1)
        cv2.imshow('band3',mask4)
        dist_transform = cv2.distanceTransform(mask4,cv2.DIST_L2,5)
        ret, mask4 = cv2.threshold(dist_transform,0.07*dist_transform.max(),255,0)
    
        cv2.imshow('band4',mask4)
        mask4 = np.array(mask4)
        
        
        
#        mask4 = np.uint8(mask4)
#        im2, contours, hierarchy = cv2.findContours(mask4, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
#        im3, newcontours, newhierarchy = cv2.findContours(mask4, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#        scr1 = cv2.drawContours(scr1, newcontours, -1, (0,255,255), 3)
#        cv2.imshow('scr1',scr1)
        
#        ncont=len(contours)
#        for i in range(ncont):
#            if(True):#len(contours[i])>100):
#                epsilon = 0.003*cv2.arcLength(contours[i], True)
#                contours[i] = cv2.approxPolyDP(contours[i], epsilon, True)
#                obstacleContoursPts += contours
            
    
            
            
    # #     cnt = contours[-5:]
    # #     scr1 = cv2.drawContours(scr, contours, 3, (0,255,0), 3)
        
#        scr2 = cv2.drawContours(scr2, contours, -1, (255,0,0), 3)    
    # #     scr1 = cv2.drawContours(scr, contours, 2, (0,0,255), 3)
    # #     scr1 = cv2.drawContours(scr, contours, 4, (0,255,255), 3)
    # #     scr1 = cv2.drawContours(scr, contours, 5, (255,0,255), 3)
    # #     scr1 = cv2.drawContours(scr, contours, 6, (255,255,0), 3)
    # #     scr1 = cv2.drawContours(scr, contours, 7, (255,255,255), 3)
    # #     scr1 = cv2.drawContours(scr, contours, 3, (0,255,0), 3)
    # #     scr1 = cv2.drawContours(scr, contours, 4, (0,255,0), 3)
    # #     scr1 = cv2.drawContours(scr, contours, 5, (0,255,0), 3)
        

        cv2.imshow('mask4',mask4)
    
        cv2.line(scr2,(0,480),(340,int(220)),(0,255,255),1)
        cv2.line(scr2,(680,480),(340,int(220)),(0,255,255),1)
        
        cv2.line(scr2,(0,int(220)),(680,int(220)),(0,0,255),2)#216  220
        
        for ind in range(0,260+20,20):
            xx = (ind)*2
            cv2.putText(scr2,str(xx),(10,480-ind), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,0),1,cv2.LINE_AA)
            cv2.line(scr2,(0,480-ind),(680,480-ind),(0,255,255),1)
    
        
        cv2.imshow('scr2',scr2)
        pathInertiaModel(obstacleContoursPts)
    #     time.sleep(0.2)
    #     print("FPS: ", 1.0 / (time.time() - start_time))
        q=cv2.waitKey(1)
        if q == 27:
            cv2.destroyAllWindows()
            break
    cv2.destroyAllWindows()
#obstacleNcounters()