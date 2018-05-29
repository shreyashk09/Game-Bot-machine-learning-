import time
from directkeys import PressKey,ReleaseKey, W, A, S, D
import random
import numpy as np

up = W
br = S
lt = A
rt = D


def straight():
    PressKey(up)
    ReleaseKey(br)
    ReleaseKey(lt)
    ReleaseKey(rt)

def left():
    if random.randrange(0,3) == 1:
        PressKey(up)
    else:
        ReleaseKey(up)
    PressKey(lt)
    ReleaseKey(rt)
    ReleaseKey(br)


def right():
    if random.randrange(0,3) == 1:
        PressKey(up)
    else:
        ReleaseKey(up)
    PressKey(rt)
    ReleaseKey(lt)
    ReleaseKey(br)
    
def reverse():
    PressKey(br)
    ReleaseKey(up)
    ReleaseKey(lt)
    ReleaseKey(rt)


def forward_left():
    PressKey(up)
    PressKey(lt)
    ReleaseKey(rt)
    ReleaseKey(br)
    
    
def forward_right():
    PressKey(rt)
    PressKey(up)
    ReleaseKey(lt)
    ReleaseKey(br)


def no_keys():

    if random.randrange(0,3) == 1:
        PressKey(W)
    else:
        ReleaseKey(W)
    ReleaseKey(A)
    ReleaseKey(S)
    ReleaseKey(D)
    



def execute():
    action = 5
    while(True):
        start = time.time()
        # action =
        #Switch action :
        if action == 0 :
            straight()
        elif action == 1 :
            left()
        elif action == 2 :
            right()
        elif action == 3 :
            forward_left()
        elif action == 4 :
            forward_right()
        elif action == 5 :
            reverse()
        else:
            no_keys()
        time.sleep(0.01)
        print("Execute FPS: ", 1.0 / (time.time() - start))

#while True:
#    execute(5)

