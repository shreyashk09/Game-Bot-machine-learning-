import numpy as np
from grabscreen import grab_screen
#import cv2
import time
from execute import execute
#import os
import threading

file_Name = "image.csv"

#while True:
#    if os.path.isfile(file_name):
#        print('File exists, moving along',starting_value)
#        starting_value += 1
#    else:
#        print('File does not exist, starting fresh!',starting_value)
#
#        break

def recScrNKey( ):

    start = time.time()
#    paused = False
    print('STARTING!!!')
#    if not paused:
#    screen = grab_screen(region=(0,40,1366,768))
    t1 = threading.Thread(target=grab_screen, args=((0,40,1366,768),))
    t1.start()
    t2 = threading.Thread(target=execute, args=())
    t2.start()
#
#
#
#    keys = getkeys()
#    if 'T' in keys:
#        if paused:
#            paused = False
#            print('unpaused!')
#            time.sleep(1)
#        else:
#            print('Pausing!')
#            paused = True
#            time.sleep(1)


recScrNKey()
