#!/usr/bin/python
import sys
import time
import math

sys.path.append('../lib/python/amd64')
import robot_interface as sdk

import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
import time

# low cmd
TARGET_PORT = 8007
LOCAL_PORT = 8082
TARGET_IP = "192.168.123.10"   # target IP address

LOW_CMD_LENGTH = 610
LOW_STATE_LENGTH = 771

# SDK MotorState&MotorCmd Sequence
DIC_SDK = {'FR_0':0, 'FR_1':1, 'FR_2':2,
         'FL_0':3, 'FL_1':4, 'FL_2':5, 
         'RR_0':6, 'RR_1':7, 'RR_2':8, 
         'RL_0':9, 'RL_1':10, 'RL_2':11 }
# RL obs sequence
DIC_RL = {'FR_0':2, 'FR_1':6, 'FR_2':10,
         'FL_0':0, 'FL_1':4, 'FL_2':8, 
         'RR_0':3, 'RR_1':7, 'RR_2':11, 
         'RL_0':1, 'RL_1':5, 'RL_2':9 }


class Walker():
    def __init__(self):
        PosStopF  = math.pow(10,9)
        VelStopF  = 16000.0
        HIGHLEVEL = 0x00
        LOWLEVEL  = 0xff
        #'FR_0':0, 'FR_1':1, 'FR_2':2,'FL_0':3, 'FL_1':4, 'FL_2':5,  'RR_0':6, 'RR_1':7, 'RR_2':8, 'RL_0':9, 'RL_1':10, 'RL_2':11 

        self.udp = sdk.UDP(LOCAL_PORT, TARGET_IP, TARGET_PORT, LOW_CMD_LENGTH, LOW_STATE_LENGTH, -1)
        #udp = sdk.UDP(8082, "192.168.123.10", 8007, 610, 771)
        self.safe = sdk.Safety(sdk.LeggedType.Aliengo)
        
        self.cmd = sdk.LowCmd()
        self.state = sdk.LowState()
        self.udp.InitCmdData(self.cmd)
        self.cmd.levelFlag = LOWLEVEL
        self.motiontime = 0

        self.targets = np.array([0.3, 0.3, -0.3, -0.3, 1.2, 1.2, 1.2, 1.2, -2.7, -2.7, -2.7, -2.7])
        self.default_pos = np.array([0.1, 0.1, -0.1, -0.1, 0.8, 1.0, 0.8, 1.0, -1.5, -1.5, -1.5, -1.5])

        self.dt = 0.02  # Control Frequency 50Hz
        self.time_record = time.time()

    def initial(self):
        while self.motiontime < 200:
            self.step()
        print("Initialize finished!")
        return


    def standup(self):
        self.initial()
        while np.max(np.abs(self.targets-self.default_pos)) > 0.01:
            self.targets -= np.clip((self.targets - self.default_pos),-0.05,0.05)
            self.step()
        while True:
            self.step()
            for i in range(12):
                print(self.state.motorState[i].q)


    def step(self):
        time.sleep(self.dt-(time.time()-self.time_record))
        self.motiontime += 1
        self.udp.Recv()
        self.udp.GetRecv(self.state)

        for dof in DIC_SDK.keys():
            self.cmd.motorCmd[DIC_SDK[dof]].q = self.targets[DIC_RL[dof]]
            self.cmd.motorCmd[DIC_SDK[dof]].dq = 0
            self.cmd.motorCmd[DIC_SDK[dof]].Kp = 400
            self.cmd.motorCmd[DIC_SDK[dof]].Kd = 10.0
            self.cmd.motorCmd[DIC_SDK[dof]].tau = 0.0
        # self.cmd.motorCmd[2].q = -2.5
        # self.cmd.motorCmd[2].dq = 0
        # self.cmd.motorCmd[2].Kp = 5
        # self.cmd.motorCmd[2].Kd = 1
        # self.cmd.motorCmd[2].tau = 0.0

        self.safe.PowerProtect(self.cmd, self.state, 9)
        self.safe.PositionLimit(self.cmd)
        
        self.udp.SetSend(self.cmd)
        self.udp.Send()
        self.time_record = time.time()


if __name__ == '__main__':
    walker = Walker()
    walker.standup()
