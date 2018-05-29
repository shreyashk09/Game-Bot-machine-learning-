# Citation: Box Of Hats (https://github.com/Box-Of-Hats )

import win32api as wapi
import time

keyList = ["\b"]
for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ 123456789,.'£$/\\":
    keyList.append(char)

def getkeys():
    while True :
    keys = []
    for key in keyList:
        if wapi.GetAsyncKeyState(ord(key)):
            keys.append(key)
    if 'T' in keys:
            if paused:
        paused = False
    print('unpaused!')
    time.sleep(1)
        else:
            print('Pausing!')
            paused = True
            time.sleep(1)
 
