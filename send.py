#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 29 20:53:36 2018

@author: shreyashkawalkar
"""

import sys
from socket import socket, AF_INET, SOCK_DGRAM
def start():
    SERVER_IP   = '127.0.0.1'
    PORT_NUMBER = 5000
    SIZE = 10
    print ("Test client sending packets to IP {0}, via port {1}\n".format(SERVER_IP, PORT_NUMBER))
    
    mySocket = socket( AF_INET, SOCK_DGRAM )
    
    while True:
            mySocket.sendto(b'cool',(SERVER_IP,PORT_NUMBER))
    sys.exit()
