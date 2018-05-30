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

mainthread1 = threading.Thread(target = obstacleNcounters, args=())
mainthread2 = threading.Thread(target = steer, args=(True))
mainthread1.start()
time.sleep(0.5)
mainthread2.start() 