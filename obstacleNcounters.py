

import numpy as np
import time
import cv2
from pathInertiaModel import pathInertiaModel
from socket import socket, gethostbyname, AF_INET, SOCK_DGRAM
from sklearn.neighbors import KNeighborsClassifier
import copy

def obstacleNcounters():
    PORT_NUMBER = 4000
    SIZE = 1024*1024
    hostName = gethostbyname( '0.0.0.0' )
    mySocket = socket( AF_INET, SOCK_DGRAM )
    mySocket.bind( (hostName, PORT_NUMBER) )
    print ("Test server listening on port {0}\n".format(PORT_NUMBER))
    
    classifier = KNeighborsClassifier(n_neighbors=1)  
    kernel = np.ones((3,3),np.uint8)
    print("running")
    prev_info_countours = []
    beg = 1

    
    while True:
        start_time = time.time()

        scr = None
        while not scr:
            (scr,addr) = mySocket.recvfrom(SIZE)
    
        scr = cv2.resize(scr, (680,480)) 
        scr = np.array(scr[220:480,:])
        cv2.imshow('scr1',scr)
        
        blank = cv2.imread('blank.png', 0)
        blank = cv2.resize(blank,(680,260))
    
        frame = cv2.GaussianBlur(scr, (5, 5), 0)
        scr = cv2.addWeighted(scr,2.5,frame,-1.5,0)
        scr = cv2.bilateralFilter(scr,9,75,75)
        
        cv2.imshow('scred',scr)
        gray = cv2.cvtColor(scr, cv2.COLOR_BGR2GRAY)
        canny = cv2.Canny(gray,50,100)
        canny = blank-canny
        cv2.imshow("blank1",canny)
        dist_transform = cv2.distanceTransform(canny,cv2.DIST_L2,5)
        ret, blank = cv2.threshold(dist_transform,0.01*dist_transform.max(),255,0)
    #     blank = cv2.dilate(blank,kernel,iterations = 2)
        blank = cv2.morphologyEx(blank, cv2.MORPH_OPEN, kernel,3)
        cv2.imshow("blankobj",blank)
        
        
        blank = np.uint8(blank)
        cntblank = cv2.imread('blank.png', 0)
        cntblank = cv2.resize(cntblank,(680,260))
        img2, contours1, hierarchy1 = cv2.findContours(blank, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        info_countours = []
        train_info_countours = []
        ind = 0
        indexes = []
        for cnt in contours1:
            if len(cnt)>5:
                indexes.append(ind)
                ind+=1
                (x,y),(MA,ma),angle = cv2.fitEllipse(cnt)
                x,y = int(x),int(y)
                mA = int(ma)//2
                color = np.mean(gray[y-mA:y+mA,x-mA:x+mA])
                y = max(min(259,y),0)
                x = max(min(679,x),0)
                if np.isnan(color):
                    color = gray[y,x]
                info = [color,y,x,angle,MA,ma]
                train_info_countours.append(info)
                
                if not beg:
                    ind_pred = classifier.predict(np.reshape(info,(-1,6)))
                    ind_pred = int(ind_pred)
                    if  len(prev_info_countours[ind_pred]) >= 15:
                        del prev_info_countours[ind_pred][0]
                    info_countours.append( prev_info_countours[ind_pred] + [info[1:3]])
                else:
                    prev_info_countours.append([info[1:3]])
    #             cv2.ellipse(scr,(x,y),(int(MA),int(ma)),angle,0,360,(0,255,255),1)
                cv2.ellipse(cntblank,(x,y),(int(MA),int(ma)),angle,0,360,50,1)
        
        
        pathInertiaModel(info_countours)
        
        
        beg = 0
        train_info_countours = np.reshape(train_info_countours,(-1,6))  
        indexes = np.reshape(indexes,(-1,1)) 
        classifier.fit(train_info_countours,indexes)
    #     scr = cv2.drawContours(scr, contours1, -1, [255,255,0], 1)
        cntblank = cv2.drawContours(cntblank, contours1, -1, 0, 1)
    #     cv2.imshow("scrobj",scr)
        cv2.imshow("blankobj2",cntblank)
        if info_countours !=[]:
            prev_info_countours = copy.deepcopy(info_countours)
        print("FPS: ", 1.0 / (time.time() - start_time))
    #     time.sleep(0.2)
    #     print("FPS: ", 1.0 / (time.time() - start_time))
        q=cv2.waitKey(1)
        if q == 27:
            cv2.destroyAllWindows()
            break
    cv2.destroyAllWindows()
#obstacleNcounters()