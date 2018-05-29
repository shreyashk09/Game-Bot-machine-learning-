# Done by Frannecklp

import numpy as np
import win32gui, win32ui, win32con, win32api
import time

keyList = ['']
def grab_screen(region=None):
    
    hwin = win32gui.GetDesktopWindow()

    if region:
            left,top,width,height = region

    else:
        width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
        height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
        left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
        top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)
    while True:
        start = time.time()
        hwindc = win32gui.GetWindowDC(hwin)
        srcdc = win32ui.CreateDCFromHandle(hwindc)
        memdc = srcdc.CreateCompatibleDC()
        bmp = win32ui.CreateBitmap()
        bmp.CreateCompatibleBitmap(srcdc, width, height)
        memdc.SelectObject(bmp)
        memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)
        
        signedIntsArray = bmp.GetBitmapBits(True)
        img = np.fromstring(signedIntsArray, dtype='uint8')
        img.shape = (height,width,4)

        srcdc.DeleteDC()
        memdc.DeleteDC()
        win32gui.ReleaseDC(hwin, hwindc)
        win32gui.DeleteObject(bmp.GetHandle())
    #   save img
        print("grab_screen 1 FPS: ", 1.0 / (time.time() - start))
    #   read keys
        if win32api.GetAsyncKeyState(ord('W')):
            if win32api.GetAsyncKeyState(ord('A')):
                actread = 3
            elif win32api.GetAsyncKeyState(ord('D')):
                actread = 4
            else:
                actread = 0
        elif win32api.GetAsyncKeyState(ord('A')):
            actread = 1
        elif win32api.GetAsyncKeyState(ord('D')):
            actread = 2
        elif win32api.GetAsyncKeyState(ord('S')):
            actread = 5
        else:
            actread = 999
        print(actread)
        print("grab_screen 2 FPS: ", 1.0 / (time.time() - start))
    # write actread

#while True:
#    grab_screen()
