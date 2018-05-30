#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 29 18:17:00 2018

@author: shreyashkawalkar
"""

import cv2
import math
from AstarAlgo import AstarAlgo

def impactZone(obstPred):
    blank = cv2.imread('blank.png',0)
    blank = cv2.resize(blank,(680,480))
    # blank = cv2.cvtColor(blank,cv2.COLOR_BGR2GRAY)
    for a,b in obstPred:
        dx = (b[0] - a[0])
        dy = (b[1] - a[1])
        dist = int(math.sqrt( dy**2 + dx**2 ))
        x,y = int(((a[0]+b[0])/2)),int((a[1]+b[1])/2)
        slp = math.degrees(dy/dx)
        #center, mj,mn,,agl,agl,color,fill
        cv2.ellipse(blank,(x,y),(6*dist,6*dist),0,0,360,100,-1)
    for a,b in obstPred:
        dx = (b[0] - a[0])
        dy = (b[1] - a[1])
        dist = int(math.sqrt( dy**2 + dx**2 ))
        x,y = int(((a[0]+b[0])/2)),int((a[1]+b[1])/2)
        slp = math.degrees(dy/dx)
        cv2.ellipse(blank,(x,y),(4*dist,3*dist),slp,0,360,50,-1)
    for a,b in obstPred:
        dx = (b[0] - a[0])
        dy = (b[1] - a[1])
        dist = int(math.sqrt( dy**2 + dx**2 ))
        x,y = int(((a[0]+b[0])/2)),int((a[1]+b[1])/2)
        slp = math.degrees(dy/dx)
        cv2.ellipse(blank,(x,y),(2*dist,1*dist),slp,0,360,0,-1)
    for a,b in obstPred:
        cv2.line(blank,(a[0],a[1]),(b[0],b[1]),255,3)
    cv2.imshow('blank',blank)
    AstarAlgo(blank)
    q = cv2.waitKey(1)
    if q == 27:
        cv2.destroyAllWindows()