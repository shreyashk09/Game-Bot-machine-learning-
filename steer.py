#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 29 18:25:24 2018

@author: shreyashkawalkar
"""

import xgboost as xgb
import numpy as np
import time
import threading
import random
from collections import deque
import sys
from socket import socket, AF_INET, SOCK_DGRAM, gethostbyname



class DQNAgent:
    def __init__(self):
        self.state_size = 15
        self.action_size = 6
        self.memory = deque(maxlen=pow(10,6))
        self.gamma = 0.8
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.bst = xgb.Booster({'nthread':4})
        self.name = 'steerDeepQL'
        self.modelname = 'xgbmodel'
        self.load_bst()
        
    
    def xgbmodel_bld(self,data,label):
        param = {'base_score' : 1, 'max_depth': 7, 'eta': 0.1, #'updater':'refresh',
            #'process_type': 'update',
            'refresh_leaf': True,
            'reshape': True,
            'reg_alpha': 3, 
            'silent': 1, 
            'objective': 'multi:softprob',     
            'num_class': 6}  
        param['nthread'] = 4
        param['eval_metric'] = 'merror'
        num_round = 500  
        dtrain = xgb.DMatrix(data, label=label)
        evallist = [(dtrain, 'train'), (dtrain, 'eval')]
        self.bst = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=3, xgb_model = self.name)
        self.save_bst()

        
    def load_bst(self):
        self.bst.load_model(self.name)
    
    def save_bst(self):
        self.bst.save_model(self.name)
        
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.bst.predict(xgb.DMatrix(state))
#         print('act_values',act_values)
        return np.argmax(act_values[0])  
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        minibatch = self.memory
#         print(len(minibatch))
        bst_target = xgb.Booster({'nthread':4})
        bst_target.load_model(self.name)
        st = []
        tg = []
        for state, action, reward, next_state, done in minibatch:

            target = bst_target.predict(xgb.DMatrix(state))##target is Q matrix
            if done:
                target[0][action] = reward
            else:
                t = self.bst.predict(xgb.DMatrix(next_state))[0]
                target[0][action] = (1-self.gamma)*reward + self.gamma * np.amax(t)
            st.append(state)
            tg.append(np.argmax(target[0]))
        self.memory.clear()
        st = np.array(st)
        st = np.reshape(st, [-1, 15])
        tg = np.array(tg)
        tg = np.reshape(tg, [-1,1])
        print(st.shape, tg.shape)
        self.xgbmodel_bld(st, tg)

        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def decl_rew(self,state):
        rew = (np.sum((np.array([1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75]) - abs(state))*np.array([10,10,10,9,9,9,8,8,7,6,5,4,3,2,1])))/175
#         print('reward',rew)
#         print('state',state)
        done = False
        if rew > 0.74:
            done = True
        return rew,done
    
    def get_nextstate(self):
        ########
        stt = np.random.randint(175, size=15)/100
        return stt
    
    def writeSocketIni(self):
        SERVER_IP   = '192.168.225.122'
        PORT_NUMBER = 6000
#        SIZE = 10
        print ("Test client sending packets to IP {0}, via port {1}\n".format(SERVER_IP, PORT_NUMBER))

        mySocket = socket( AF_INET, SOCK_DGRAM )
        return mySocket

        
    def readSocketIni(self):
        PORT_NUMBER = 5000
#        SIZE = 10
        hostName = gethostbyname( '0.0.0.0' )
        mySocket = socket( AF_INET, SOCK_DGRAM )
        mySocket.bind( (hostName, PORT_NUMBER) )
        return mySocket
    
    def stateSocketIni(self):
        PORT_NUMBER = 7000
#        SIZE = 1024
        hostName = gethostbyname( '127.0.0.1' )
        mySocket = socket( AF_INET, SOCK_DGRAM )
        mySocket.bind( (hostName, PORT_NUMBER) )
        return mySocket
    
def steer(GameOn):
    
    state_size = 15
#    action_size = 6
    agent = DQNAgent()
    state = None
    mysocket = agent.stateSocketIni()
    while not state:
        (state,addr) = mysocket.recvfrom(1024)
#    state = agent.get_nextstate()
#    count = 0
#    st = []
#    tg = []
    mysocket1 = agent.readSocketIni()
    mysocket2 = agent. writeSocketIni()
    mysocket2.sendto(0,('192.168.225.146',6000))
    mysocket2.sendto(0,('192.168.225.146',6000))
    bst_target = xgb.Booster({'nthread':4})
    bst_target.load_model(agent.name)
    while GameOn:
#        start = time.time()
        (read_Action,addr) = mysocket1.recvfrom(10)
        state = np.reshape(state, [1, state_size])
        action = agent.act(state)
#        stract = str(action) 
        mysocket2.sendto(action,('192.168.225.146',6000))
        mysocket2.sendto(action,('192.168.225.146',6000))
        
        reward, done = agent.decl_rew(state)
#         reward = reward 
        (next_state,addr) = mysocket.recvfrom(1024)#agent.get_nextstate()
        if next_state == [] or read_Action==999:
            continue;
#        count += 1
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, read_Action, reward, next_state, done)
#         target = bst_target.predict(xgb.DMatrix(state))
#         if done:
#             target[0][action] = reward
#         else:
#             t = agent.bst.predict(xgb.DMatrix(next_state))[0]
#             target[0][action] = (1-agent.gamma)*reward + agent.gamma * np.amax(t)
#         st.append(state)
#         tg.append(np.argmax(target[0]))
        
#         if count % 250 == 0:
#             count = 0
#             st = np.array(st)
#             st = np.reshape(st, [-1, 15])
#             tg = np.array(tg)
#             tg = np.reshape(tg, [-1,1])
#             agent.xgbmodel_bld(st, tg)
#             if agent.epsilon > agent.epsilon_min:
#                 agent.epsilon *= agent.epsilon_decay
#             st = []
#             tg = []
#             bst_target = None
#             bst_target = xgb.Booster({'nthread':4})
#             bst_target.load_model(agent.name)
#             print("success")
        state = next_state
        if done and len(agent.memory) >= 50 or len(agent.memory) >= 500 :
            print("replaying at {} xgboost".format(len(agent.memory)))
            t1 = threading.Thread(target=agent.replay, args=())
            t1.start()
#         time.sleep(0.03)
#        print("FPS: ", 1.0 / (time.time() - start))
        
