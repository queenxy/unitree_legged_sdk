#!/usr/bin/python
import sys
import time
import math
import pickle as pkl
import io

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
DIC_RL = {'FR_0':3, 'FR_1':4, 'FR_2':5,
         'FL_0':0, 'FL_1':1, 'FL_2':2, 
         'RR_0':9, 'RR_1':10, 'RR_2':11, 
         'RL_0':6, 'RL_1':7, 'RL_2':8 }

def get_rotation_matrix_from_rpy(rpy):
    """
    Get rotation matrix from the given quaternion.
    Args:
        q (np.array[float[4]]): quaternion [w,x,y,z]
    Returns:
        np.array[float[3,3]]: rotation matrix.
    """
    r, p, y = rpy
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(r), -math.sin(r)],
                    [0, math.sin(r), math.cos(r)]
                    ])

    R_y = np.array([[math.cos(p), 0, math.sin(p)],
                    [0, 1, 0],
                    [-math.sin(p), 0, math.cos(p)]
                    ])

    R_z = np.array([[math.cos(y), -math.sin(y), 0],
                    [math.sin(y), math.cos(y), 0],
                    [0, 0, 1]
                    ])

    rot = np.dot(R_z, np.dot(R_y, R_x))
    return rot

class CPU_Unpickler(pkl.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)


class Walker():
    def __init__(self):
        PosStopF  = math.pow(10,9)
        VelStopF  = 16000.0
        HIGHLEVEL = 0x00
        LOWLEVEL  = 0xff
        #'FR_0':0, 'FR_1':1, 'FR_2':2,'FL_0':3, 'FL_1':4, 'FL_2':5,  'RR_0':6, 'RR_1':7, 'RR_2':8, 'RL_0':9, 'RL_1':10, 'RL_2':11 

        self.device = "cpu"
        self.load_policy("checkpoints/")

        self.udp = sdk.UDP(LOCAL_PORT, TARGET_IP, TARGET_PORT, LOW_CMD_LENGTH, LOW_STATE_LENGTH, -1)
        #udp = sdk.UDP(8082, "192.168.123.10", 8007, 610, 771)
        self.safe = sdk.Safety(sdk.LeggedType.Aliengo)
        
        self.cmd = sdk.LowCmd()
        self.state = sdk.LowState()
        self.udp.InitCmdData(self.cmd)
        self.cmd.levelFlag = LOWLEVEL
        self.motiontime = 0

        # self.targets = np.array([0.3, 0.3, -0.3, -0.3, 1.2, 1.2, 1.2, 1.2, -2.7, -2.7, -2.7, -2.7])
        # self.default_pos = np.array([0.1, 0.1, -0.1, -0.1, 0.8, 1.0, 0.8, 1.0, -1.5, -1.5, -1.5, -1.5])
        self.targets = np.array([0.3, 1.2, -2.7, -0.3, 1.2, -2.7, 0.3, 1.2, -2.7, -0.3, 1.2, -2.7])
        self.default_pos = np.array([0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5])

        self.dt = 0.02  # Control Frequency 50Hz
        self.time_record = time.time()  # Record the time begin to get obs and calculate action

        # during standup, set Kp,Kd large numbers; during walking, set to smaller numbers
        self.Kp = 80
        self.Kd = 2.0

        self.past_action = np.zeros(12)
        self.past_two_action = np.zeros(12)
        self.dof_pos = np.zeros(12)
        self.dof_vel = np.zeros(12)
        self.commands = np.array([1.5,0.0,0.0,0.0,3.0,0.5,0.0,0.0,0.5,0.12,0.0,0.0,0.32,0.42803,2.1376e-4])
        self.clock_inputs = np.zeros(4)
        self.gait_indices = np.zeros(1)

        self.obs_history = np.zeros(self.num_obs_history).astype(np.float32)

        

    def initial(self):
        self.Kp = 20
        self.Kd = 1.0
        while self.motiontime < 200:
            self.time_record = time.time()
            self.step()
            time.sleep(self.dt-(time.time()-self.time_record))        
        print("Initialize finished!")
        return


    def standup(self):
        self.initial()
        self.Kp = 400
        self.Kd = 10.0
        while np.max(np.abs(self.targets-self.default_pos)) > 0.01:
            self.time_record = time.time()
            self.targets -= np.clip((self.targets - self.default_pos),-0.05,0.05)
            self.step()
            time.sleep(self.dt-(time.time()-self.time_record))
        print("Stand up Successfully!")
        while self.motiontime < 500:
            self.time_record = time.time()
            self.udp.Recv()
            self.udp.GetRecv(self.state)
            # for i in range(12):
            #     print(self.state.motorState[i].q)
            # self.get_obs(self.state)
            self.step()
            time.sleep(self.dt-(time.time()-self.time_record))

        self.step()

        return


    def step(self):
        self.motiontime += 1
        for dof in DIC_SDK.keys():
            self.cmd.motorCmd[DIC_SDK[dof]].q = self.targets[DIC_RL[dof]]
            self.cmd.motorCmd[DIC_SDK[dof]].dq = 0
            self.cmd.motorCmd[DIC_SDK[dof]].Kp = self.Kp
            self.cmd.motorCmd[DIC_SDK[dof]].Kd = self.Kd
            self.cmd.motorCmd[DIC_SDK[dof]].tau = 0.0

        self.safe.PowerProtect(self.cmd, self.state, 9)
        self.safe.PositionLimit(self.cmd)
        
        self.udp.SetSend(self.cmd)
        self.udp.Send()
    
    def load_policy(self, logdir):
        self.body = torch.jit.load(logdir + 'body_latest.jit')
        self.adaptation_module = torch.jit.load(logdir + 'adaptation_module_latest.jit')
        with open(logdir+"parameters.pkl", 'rb') as file:
            pkl_cfg = CPU_Unpickler(file).load()
            self.cfg = pkl_cfg["Cfg"]

        # The first forward will cost 0.1s, run here to keep control frequency
        latent = self.adaptation_module.forward(torch.zeros(2100))
        
        self.action_scale = self.cfg["control"]["action_scale"]
        self.hip_scale_reduction = self.cfg["control"]["hip_scale_reduction"]

        self.pos_scale = self.cfg["obs_scales"]["dof_pos"]
        self.vel_scale = self.cfg["obs_scales"]["dof_vel"]
        self.num_commands = self.cfg["commands"]["num_commands"]
        self.cmd_scale = np.array([self.cfg["obs_scales"]["lin_vel"],self.cfg["obs_scales"]["lin_vel"],
                                   self.cfg["obs_scales"]["ang_vel"],self.cfg["obs_scales"]["body_height_cmd"],1, 1, 1, 1, 1,
                                   self.cfg["obs_scales"]["footswing_height_cmd"], self.cfg["obs_scales"]["body_pitch_cmd"],
                                    self.cfg["obs_scales"]["body_roll_cmd"], self.cfg["obs_scales"]["stance_width_cmd"],
                                    self.cfg["obs_scales"]["stance_length_cmd"], self.cfg["obs_scales"]["aux_reward_cmd"], 1, 1, 1, 1, 1, 1
                                   ])[:self.num_commands]
        self.clip_action = self.cfg["normalization"]["clip_actions"]

        self.obs_history_length = self.cfg["env"]["num_observation_history"]
        self.num_obs = self.cfg["env"]["num_observations"]
        self.num_obs_history = self.obs_history_length * self.num_obs
        print(self.obs_history_length,self.num_obs,self.num_obs_history)
            
    def run(self):
        self.Kp = self.cfg["control"]["stiffness"]['joint']
        self.Kd = self.cfg["control"]["damping"]['joint']
        # self.gait_indices = np.zeros(1)
        # desire_pos = []

        while True:
            self.time_record = time.time()

            # Get state from Motor
            self.udp.Recv()
            self.udp.GetRecv(self.state)
            # print(time.time()-self.time_record)
            # Calculate Obs from state
            self.get_obs(self.state)
            # print(time.time()-self.time_record)
            # Calculate action
            self.get_action()
            # print(time.time()-self.time_record)
            # desire_pos.append(self.targets)
            # Step env
            self.step()
            # print(time.time()-self.time_record)
            # Keep frequency of dt
            # if time.time()-self.time_record < self.dt:
            time.sleep(self.dt-(time.time()-self.time_record))


    def get_obs(self,low_state):
        rpy = np.array(low_state.imu.rpy)
        R = get_rotation_matrix_from_rpy(rpy)

        for dof in DIC_SDK.keys():
            self.dof_pos[DIC_RL[dof]]= low_state.motorState[DIC_SDK[dof]].q
            self.dof_vel[DIC_RL[dof]] = low_state.motorState[DIC_SDK[dof]].dq

        gravity_vec = np.dot(R.T, np.array([0, 0, -1]))

        obs = np.concatenate(
            (
                gravity_vec,
                self.commands * self.cmd_scale,
                (self.dof_pos - self.default_pos) * self.pos_scale,
                self.dof_vel * self.vel_scale,
                self.past_action,
            ),
        )

        if self.cfg["env"]["observe_two_prev_actions"]:
            obs = np.concatenate((obs,
                            self.past_two_action))

        if self.cfg["env"]["observe_clock_inputs"]:
            frequencies = self.commands[4]
            phases = self.commands[5]
            offsets = self.commands[6]
            if self.num_commands == 8:
                bounds = 0
                durations = self.commands[ 7]
            else:
                bounds = self.commands[7]
                durations = self.commands[8]
            self.gait_indices = (self.gait_indices + self.dt * frequencies)%1.0

            if "pacing_offset" in self.cfg["commands"] and self.cfg["commands"]["pacing_offset"]:
                self.foot_indices = [self.gait_indices + phases + offsets + bounds,
                                    self.gait_indices + bounds,
                                    self.gait_indices + offsets,
                                    self.gait_indices + phases]
            else:
                self.foot_indices = [self.gait_indices + phases + offsets + bounds,
                                    self.gait_indices + offsets,
                                    self.gait_indices + bounds,
                                    self.gait_indices + phases]
            self.clock_inputs[0] = math.sin(2 * np.pi * self.foot_indices[0])
            self.clock_inputs[1] = math.sin(2 * np.pi * self.foot_indices[1])
            self.clock_inputs[2] = math.sin(2 * np.pi * self.foot_indices[2])
            self.clock_inputs[3] = math.sin(2 * np.pi * self.foot_indices[3])

            obs = np.concatenate((obs,
                            self.clock_inputs))

        obs = obs.astype(np.float32)
        # print("Obs:",obs)
        self.obs_history = np.concatenate((self.obs_history[self.num_obs:],obs))
        # print(self.obs_history.shape)

    def get_action(self):
        # t = time.time()
        obs_history_t = torch.from_numpy(self.obs_history)
        # print(time.time()-t)
        # print(obs_history_t)
        latent = self.adaptation_module.forward(obs_history_t)
        # print(time.time()-t)
        # print(torch.cat((obs_history_t, latent)).dtype)
        action = self.body.forward(torch.cat((obs_history_t, latent), dim=-1))
        # print(time.time()-t)
        
        # print("Action:", action)
        action = torch.clamp(action, -self.clip_action, self.clip_action).cpu().detach().numpy()
        self.past_two_action = self.past_action
        self.past_action = action
        
        scaled_action = self.action_scale * action
        scaled_action[0] *= self.hip_scale_reduction
        scaled_action[3] *= self.hip_scale_reduction
        scaled_action[6] *= self.hip_scale_reduction
        scaled_action[9] *= self.hip_scale_reduction
        
        self.targets = scaled_action + self.default_pos
        # print("Action:",self.targets)
        return self.targets


if __name__ == '__main__':
    walker = Walker()
    walker.standup()
    walker.run()
