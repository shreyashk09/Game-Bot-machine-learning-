#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 29 18:29:12 2018

@author: shreyashkawalkar
"""
import keras.models
import numpy as np
from impactZone import impactZone

def loadModel():
    loaded_model = None
    with open("pathInertia.json", "r") as json_file:
        loaded_model_json = json_file.read()
        loaded_model = keras.models.model_from_json(loaded_model_json)
        loaded_model.load_weights("pathInertia.h5")
        print("Model loaded from disk")
    return loaded_model

def pathInertiaModel(obstacleContoursPts):
    obstPred = []
    loaded_model = loadModel()
    for pts in obstacleContoursPts:#pts---series of 0bj path
        pred = loaded_model.predict(np.array([pts]))#np.array([[[2,1],[2,1],[2,1],[2,1],[2,1],[2,1],[2,1],[2,1],[2,1]]]))
        obstPred.append([pts[-1],pred])
    impactZone(obstPred)