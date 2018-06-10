#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 15:56:15 2018

@author: shreyashkawalkar
"""

import threading 
import time
from obstacleNcounters import obstacleNcounters
from steer import steer
from impactZone import impactZone

pathInertia = None
preImpactZoneImage = None
postImpactZone = None

mainthread1 = threading.Thread(target = obstacleNcounters, args=())
subThread = threading.Thread(target = impactZone , args=())
mainthread1.start()
subThread.start()

time.sleep(0.1)

mainthread2 = threading.Thread(target = steer, args=(True))
mainthread2.start() 