#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 29 18:29:12 2018

@author: shreyashkawalkar
"""
import keras.models
import numpy as np
from impactZone import impactZone
import math

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
    print("para cal")
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
    print("para cal")
    return setpara

def pointV(orgP, predP):
    y = orgP[1]*(math.sin(predP[1]))
    x = orgP[0]*(math.cos(predP[1]))
    if not predP[2]:
        y=-1*y
    return [x, y]
        
    
def pathInertiaModel(obstacleContoursPts):
    obstPred = []
    loaded_model = loadModel()
    for pts in obstacleContoursPts:#pts---series of 0bj path
        pointsParameter = calcParameters(pts)
        pred = loaded_model.predict(np.reshape(pointsParameter,(-1,9,3)))#np.array([[[2,1],[2,1],[2,1],[2,1],[2,1],[2,1],[2,1],[2,1],[2,1]]]))
        obstPred.append([pts[-1],pointV(pts[-1],pred[0])])
#    print("pathInertiaMotion::",np.array(obstPred).shape)
    print("done")
    return obstPred
#    impactZone(obstPred)