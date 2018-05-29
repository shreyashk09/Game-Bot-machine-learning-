#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 29 18:17:00 2018

@author: shreyashkawalkar
"""

import cv2
import numpy as np

def impactZone():
    path = [[0, 10], [1, 9], [2, 9], [3, 10], [4, 9], [5, 10], [6, 9], [7, 9], [8, 10], [9, 10], [10, 11], [11, 10], [12, 10], [13, 11],[14, 11], [15, 11], [16, 10], [17, 10], [18, 9], [19, 10]]
    blank = cv2.imread("blank.png",0)
    blank = cv2.resize(blank,(20,20))
    # path = np.array(path)
    # epsilon = 0.1*cv2.arcLength(path, True)
    # path = cv2.approxPolyDP(path, epsilon, False)
    # print(path)
    pnts = []
    for i in range(1,16):
        pts = [np.mean(path[0:15*(i)][0]),np.mean(path[0:15*(i)][1])]
        pnts.append(pts)
    #     cv2.line(blank,(path[i][1],path[i][0]),(path[i+1][1],path[i+1][0]),0)
    #     pnts = np.mean(path[i:i+15][0])
    print(pnts)
    blank = cv2.resize(blank,(680,480))
    cv2.imshow("blank",blank)
    q = cv2.waitKey(1)
    if q == 27:
        cv2.destroyAllWindows()