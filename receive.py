#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 29 20:52:36 2018

@author: shreyashkawalkar
"""


from socket import socket, gethostbyname, AF_INET, SOCK_DGRAM
import sys
from send import start
import time
PORT_NUMBER = 7000
SIZE = 1024

hostName = gethostbyname( '127.0.0.1' )

mySocket = socket( AF_INET, SOCK_DGRAM )
mySocket.bind( (hostName, PORT_NUMBER) )

print ("Test server listening on port {0}\n".format(PORT_NUMBER))
#start()
while True:
        start_time = time.time()
        (data,addr) = mySocket.recvfrom(SIZE)
        print (data)
        print("FPS: ", 1.0 / (time.time() - start_time))
sys.ext()