#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 29 18:10:17 2018

@author: shreyashkawalkar
"""

import random
import numpy as np
import math
from steer import steer

def slopes(pathpts):
    slopeAngle = []
    for i in range(1,16):
        pts = [np.mean(pathpts[15*(i-1):15*(i)][0]),np.mean(pathpts[15*(i-1):15*(i)][1])]
        slopeAngle.append(math.atan2(340-pts[1],pts[0]-240) - 1.57)      
    return slopeAngle

def AstarAlgo(obsTopo):
    targ = 0
    shape = [20,20]
    close = [[0 for i in range(shape[1])] for j in range(shape[0])]
    start = [19,10]
    val = [[random.randint(1, 255) for i in range(shape[1])]for j in range(shape[0])]
    gval = [[999999 for i in range(shape[1])]for j in range(shape[0])]
    hval = [[i for i in range(shape[1])]for j in range(shape[0])]
    fval = [[999999 for i in range(shape[1])]for j in range(shape[0])]
    target = [[0,x] for x in range(shape[1])]
    parent = [[0 for i in range(shape[1])]for j in range(shape[0])]
    gval[start[0]][start[1]] = 0
    parent[start[0]][start[1]] = -1
    
    
    def neigh(pos):
        ngh = []
        if(pos[1]<=1 or pos[1]>=shape[1]-2):
            return ngh
    #     ngh.append([pos[0], pos[1]+1])
    #     ngh.append([pos[0],pos[1]-1])
        ngh.append([pos[0]-1,pos[1]])
        ngh.append([pos[0]-1,pos[1]+1])
        ngh.append([pos[0]-1, pos[1]-1])
        return ngh
    def cost(par,child):
        return (val[child[0]][child[1]] - val[par[0]][par[1]])
    
    reqpath = start
    minf = 0
    
    def derive(node):
        print(node,"->",end = " ")
#        path2target.append()
        while node!=start and node!=0:
            node = parent[node[0]][node[1]]
            print(node,"->",end=" ")
            
    def path(node):
        if node in target:
    #         if fval[node[0]][node[1]]<minf:
    #             minf = fval[node[0]][node[1]]
    #             reqpath = node
            derive(node)
            print("target: fval",fval[node[0]][node[1]])
            return
        fsort = []
        nsort=[]
        neighlist = neigh(node)
        close[node[0]][node[1]] = 1
        if len(neighlist) == 0:
            print("border")
            return
    #     print(neighlist)
        k=0
        for ngh in neighlist:
            cst = cost(node,ngh)
    #         print("close",ngh," ",close[ngh[0]][ngh[1]])
            if (close[ngh[0]][ngh[1]]==0):
                if cst + gval[node[0]][node[1]] < gval[ngh[0]][ngh[1]] :
                    gval[ngh[0]][ngh[1]] = cst + gval[node[0]][node[1]]
                    parent[ngh[0]][ngh[1]] = node
                    k+=1
    #                 print("close",ngh," ",close[ngh[0]][ngh[1]])
                fval[ngh[0]][ngh[1]] = hval[ngh[0]][ngh[1]] + gval[ngh[0]][ngh[1]]
                fsort.append(fval[ngh[0]][ngh[1]])
                nsort.append(ngh)
    #     print(k)
        srt = [x for _,x in sorted(zip(fsort,nsort))]
        
        for nd in srt:
                path(nd)
                   
    steer(slopes(path([480,340])))