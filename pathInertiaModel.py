#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 29 18:29:12 2018

@author: shreyashkawalkar
"""

import keras.models
def pathInertiaModel():
    with open("pathInertia.json", "r") as json_file:
        loaded_model_json = json_file.read()
        loaded_model = keras.models.model_from_json(loaded_model_json)
        loaded_model.load_weights("pathInertia.h5")
        print("Model loaded from disk")
        loaded_model.summary()
    
    for i,j in zip(train_data,train_target):
    #    print("input")
    #    print(i)
    #    print("predict::")
    #    print(j)
        print(loaded_model.predict(np.array([i])))#np.array([[[2,1],[2,1],[2,1],[2,1],[2,1],[2,1],[2,1],[2,1],[2,1]]]))
