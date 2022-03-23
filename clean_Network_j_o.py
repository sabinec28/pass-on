#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 13:15:23 2021

@author: sabinechavin
"""

import time
import numpy as np
import math
import os
from scipy import stats, io
import clean_SetUp_j_o as SN #code for setting up initial network configuration and dealing with trajectory stuff
from pathlib import Path
import importlib
import scipy

importlib.reload(SN)

#importing just for visualizing
import matplotlib.pyplot as plt  

    
class EBC_info:   
    """
    @params
    n_EBC = num of egocentric boundary cells
    base_input_EBC = base input values for EBC's
    max_dis_threshold = maximum distance from wall that EBC's will fire
    rand_weights_max = maximum value for initial random weights in weight matrix
    inhib_weight = scaling factor for lateral inhibition
    epsilon = parameter in Hebbian learning that controls strength of EBC input
    """
    def __init__(self, n_EBC, base_input_EBC, max_dis_threshold,rand_weights_max_EBC, inhib_weight_EBC, epsilon_EBC, rate):
        self.n_EBC = n_EBC
        self.base_input = base_input_EBC
        self.max_dis_threshold = max_dis_threshold
        self.rand_weights_max = rand_weights_max_EBC
        self.inhib_weight = inhib_weight_EBC
        self.learn_rate = rate
        self.epsilon = epsilon_EBC
        
        
class HD_info:
    """
    @params
    n_HD = num of head direction cells
    base_input_HD = base input values for HD's
    rand_weights_max = maximum value for initial random weights in weight matrix
    inhib_weight = scaling factor for lateral inhibition
    epsilon = parameter in Hebbian learning that controls strength of HD input
    """
    def __init__(self, n_HD, base_input_HD, rand_weights_max_HD, inhib_weight_HD, epsilon_HD, rate):
        self.n_HD = n_HD
        self.base_input = base_input_HD
        self.rand_weights_max = rand_weights_max_HD
        self.inhib_weight = inhib_weight_HD
        self.epsilon = epsilon_HD        
        self.learn_rate = rate
        
class EBCxHD_info:
    
    def __init__(self,n_EBCxHD, max_dis_threshold_EBCxHD, base_input_EBCxHD,rand_weights_max_EBCxHD, inhib_weight_EBCxHD, epsilon_EBCxHD, rate,
                 place_bias, bias_size):
        self.n_EBCxHD = n_EBCxHD
        self.max_dis_threshold = max_dis_threshold_EBCxHD
        self.base_input = base_input_EBCxHD
        self.rand_weights_max = rand_weights_max_EBCxHD
        self.inhib_weight = inhib_weight_EBCxHD
        self. epsilon =  epsilon_EBCxHD
        self.learn_rate = rate
        self.place_bias = place_bias ##boolean to decide if to include place bias or not
        self.bias_size = bias_size
        self.list_loc = []
        
class place_info:
    def __init__(self,n_place,radius, dim):
        self.n_place = n_place
        self.radius = radius
        self.dim = dim
        
       
class ABC_info:
    
    def __init__(self,n_ABC,base_input_ABC, inhib_weight_ABC):
        self.n_ABC = n_ABC
        self.base_input = base_input_ABC
        self.inhib_weight = inhib_weight_ABC
        
        
class network:

    ## Create an Network object to hold model parameters and run simulations.
    
    def __init__(self, ABC_info,EBCxHD_info,EBC_info,HD_info,dt,tau, manual_weights, learn, filename, place_info):
        self.n_ABC = ABC_info.n_ABC #number of (putative) ABCs in the output layer (used 16 originally just trying this out)
        self.n_EBCxHD = EBCxHD_info.n_EBCxHD #number of (putative or hard coded EBCs in the middle layer)
        self.n_EBC = EBC_info.n_EBC #number of EBCs
        self.n_HD = HD_info.n_HD #number of HDs
        self.n_place = place_info.n_place
        
        self.dt = dt #simulation resolution in miliseconds 
        self.tau = tau ## # neuron time constant in ms
        self.EBC_info = EBC_info
        self.HD_info = HD_info
        self.EBCxHD_info = EBCxHD_info
        self.ABC_info = ABC_info
        self.manual_weights = manual_weights # boolean toggle, TRUE if manual weights, FALSE if not manual weights
        self.learn = learn
        self.filename = filename
        self.place_info = place_info
    
        self.load_trajectory()
        self.precalc_ego_walldis()
        if manual_weights: ##so far only hardcoding in the transform 1 weights
            self.setup_manual_weights()
        else:
            self.setup_weights()
        self.setup_receptive_fields()
        self.init_w_EBCxHD_EBC = self.w_EBCxHD_EBC.copy()
        self.init_w_EBCxHD_HD = self.w_EBCxHD_HD.copy()
        self.init_w_ABC_EBCxHD = self.w_ABC_EBCxHD.copy()
        
        
       
        """
        ##variable to store weights over all time to see evolution/stability/convergrence
        self.all_w_EBCxHD_HD =  np.zeros((self.time_ind,self.init_w_EBCxHD_HD.shape[0],self.init_w_EBCxHD_HD.shape[1]))
        self.all_w_EBCxHD_EBC = np.zeros((self.time_ind,self.init_w_EBCxHD_EBC.shape[0],self.init_w_EBCxHD_EBC.shape[1]))
        self.all_w_EBCxHD_HD[0] = self.init_w_EBCxHD_HD
        self.all_w_EBCxHD_EBC[0] =  self.init_w_EBCxHD_EBC
        
        """
        
    ######################################################################################
    ############################Trajectory Data Loading/Precalculation + Set up Network#######################
    ############################################################################################
    
    def load_trajectory(self):
       ##  'Trajectory_Data_test_full_HD.mat' original file
        #Load in trajectory (see function defintion for notes on how to update)
        [self.x,self.y,self.x_ind,self.y_ind, self.headdir_ind, self.x_of_bins,self.y_of_bins, self.spatial_scale, self.boundaries, self.time_ind, self.traj_filename] = SN.get_trajectory(self.filename, self.dt)
        pos = np.where(self.headdir_ind>0,self.headdir_ind,self.headdir_ind+360)
        self.headdir_ind = pos
    #Trying new implementation for now.  Precalculate these values.  Add ability to save output so don't have to rerun and add check to see if file already exists
    #As this values don't change as long as traj is the same.
    
    def precalc_ego_walldis(self):
        if self.filename == 'Trajectory_Data_test_full_HD.mat':
            precalc_file = Path("test_1_n=48.mat")
        if self.filename == 'Trajectory_Data_test2_full_HD.mat':
            precalc_file = Path("second_test.mat")
        if self.filename == 'Trajectory_Data_test3_full_HD.mat':
            precalc_file = Path("third_test.mat")
        if precalc_file.is_file():
           print('Precalculations already done')
           unpack = io.loadmat(precalc_file)
           self.ego_walldis = unpack['ego_walldis']
           
        else:
            print('Doing precalculations based on trajectory data')
            self.ego_walldis = SN.get_egostates(self.x_ind,self.y_ind,self.headdir_ind, self.x_of_bins,self.y_of_bins,self.boundaries,self.spatial_scale,self.time_ind, self.traj_filename,self.EBC_info.max_dis_threshold)

    ## helper function takes as input number in output layer, input layer and rand_weight_max
    def weights(self, out_cells, in_cells, rand_max):
        w = np.reshape(stats.uniform.rvs(0,rand_max, out_cells*in_cells),(in_cells,out_cells))
        return w
    
    def normalize_weights(self,w, val):
    
        cols = len(w[0])
        rows = len(w)

        for c in range(cols):
            s = 0
            for r in range(rows):
                s += w[r][c]
            k = 0.008/s
            for r in range(rows):
                w[r][c] = w[r][c]*k
        return w
    
    def setup_weights(self):
        #Setup initial random weights between layers being simulated
        self.w_EBCxHD_HD = self.weights(self.n_EBCxHD, self.n_HD, self.HD_info.rand_weights_max)
        self.w_EBCxHD_EBC = self.weights(self.n_EBCxHD, self.n_EBC, self.EBC_info.rand_weights_max)
        self.w_ABC_EBCxHD = self.weights(self.n_ABC, self.n_EBCxHD, self.EBCxHD_info.rand_weights_max)
        
        self.w_EBCxHD_HD = self.normalize_weights(self.w_EBCxHD_HD, self.EBC_info.epsilon)
        self.w_EBCxHD_EBC = self.normalize_weights(self.w_EBCxHD_EBC,self.HD_info.epsilon)
        
        ## this is a placeholder, not using this weight matrix at all yet
        self.w_EBCxHD_EBC_HD = self.weights(self.n_EBCxHD, self.n_HD*self.n_EBC, self.EBC_info.rand_weights_max)

        self.w_EBCxHD_place= self.weights(self.n_EBCxHD, self.n_place, self.EBCxHD_info.rand_weights_max)
        self.w_EBCxHD_place= self.normalize_weights(self.w_EBCxHD_place, self.EBC_info.epsilon)
      
    def setup_receptive_fields(self):
        #Setup receptive fields
        self.HD_fields = SN.setup_HD_fields(self.n_HD)
        self.EBC_fields = SN.setup_EBC_fields(self.n_EBC, self.EBC_info.max_dis_threshold)
        self.EBCxHD_fields = SN.setup_EBCxHD_fields(self.n_EBCxHD, self.EBCxHD_info.max_dis_threshold)
       
        self.place_grid = SN.setup_place_grid(self.place_info.n_place, self.place_info.dim, self.place_info.radius)
    
    ## hard coding weights
    def setup_manual_weights(self):
        
        self.w_EBCxHD_HD = self.weights(self.n_EBCxHD, self.n_HD, self.HD_info.rand_weights_max)
        self.w_EBCxHD_HD = self.normalize_weights(self.w_EBCxHD_HD,0.008)
        
        self.w_EBCxHD_EBC = self.weights(self.n_EBCxHD, self.n_EBC, self.EBC_info.rand_weights_max)
        self.w_EBCxHD_EBC = self.normalize_weights(self.w_EBCxHD_EBC,0.008)
        
        #self.w_ABC_EBCxHD = self.weights(self.n_ABC, self.n_EBCxHD, self.EBCxHD_info.rand_weights_max)
        #self.w_EBCxHD_EBCxHD = self.weights(self.n_EBCxHD, self.n_EBCxHD, self.EBC_info.rand_weights_max)
        #self.w_EBCxHD_EBCxHD = self.normalize_weights(self.w_EBCxHD_EBCxHD,0.008)
        
        
        self.w_ABC_EBCxHD = self.weights(self.n_ABC, self.n_EBCxHD, 0.008)
        ## all Forward wall cells
        
        """
         ## all Forward wall cells hard coding
        F_cells = np.array([0,7,10,13])
        L_cells = np.array([1,4,11,14])
        R_cells = np.array([3,6,9,12])
        B_cells = np.array([2,5,8,15])
        w_ABC = 0.007
        self.w_ABC_EBCxHD[F_cells,0]=w_ABC
        self.w_ABC_EBCxHD[L_cells,1]=w_ABC
        self.w_ABC_EBCxHD[R_cells,2]=w_ABC
        self.w_ABC_EBCxHD[B_cells,3]=w_ABC
        """
        
        ## initialize toeplitz matrix of zeros/ones in HD cells
        ylist = np.zeros(self.n_HD)
        xlist = np.zeros(self.n_EBCxHD)
        ylist[0] = 1
        i = self.n_HD
        while i < self.n_EBCxHD:
            xlist[i] = 1
            i += self.n_HD
            
        x = scipy.linalg.toeplitz(ylist,xlist)
        x = x*self.EBCxHD_info.epsilon
        self.w_EBCxHD_HD = self.weights(self.n_EBCxHD, self.n_HD, self.EBCxHD_info.epsilon/(2*self.n_HD))
        self.w_EBCxHD_HD = self.w_EBCxHD_HD + x
        
        ylist = np.zeros(self.n_EBC)
        xlist = np.zeros(self.n_EBCxHD)
        ylist[0] = 1
        i = self.n_EBC
        while i < self.n_EBCxHD:
            xlist[i] = 1
            i += self.n_EBC
            
        x = scipy.linalg.toeplitz(ylist,xlist)
        x = x*self.EBCxHD_info.epsilon
        self.w_EBCxHD_EBC = self.weights(self.n_EBCxHD, self.n_EBC, self.EBCxHD_info.epsilon/(2*self.n_EBC))
        self.w_EBCxHD_EBC = self.normalize_weights(self.w_EBCxHD_EBC,0.008)
        self.w_EBCxHD_EBC = self.w_EBCxHD_EBC + x
        
        if self.n_EBC == 8:
            if self.n_EBCxHD ==32:
                x = scipy.linalg.toeplitz([1,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0])
                w = self.weights(self.n_EBCxHD, self.n_EBC,  0.45*self.HD_info.epsilon )
                t = np.where(x>0, 0.45*self.HD_info.epsilon,w)
                self.w_EBCxHD_EBC = t
            if self.n_EBCxHD == 64:
                x = scipy.linalg.toeplitz([1,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0])
                w = self.weights(self.n_EBCxHD, self.n_EBC, 0.002)
                t = np.where(x>0,0.003,w)
                self.w_EBCxHD_EBC = t
            
        
        if self.n_HD == 32:
            if self.n_EBCxHD == 256:
                self.w_EBCxHD_HD = self.weights(self.n_EBCxHD, self.n_HD, 0.002)
                big_w =  0.8*self.HD_info.epsilon  
                self.w_EBCxHD_HD[0][0:8] = big_w
                self.w_EBCxHD_HD[1][8:16] = big_w
                self.w_EBCxHD_HD[2][16:24] = big_w
                self.w_EBCxHD_HD[3][24:32] = big_w
                self.w_EBCxHD_HD[4][32:40] = big_w
                self.w_EBCxHD_HD[5][40:48] = big_w
                self.w_EBCxHD_HD[6][48:56] = big_w
                self.w_EBCxHD_HD[7][56:64] = big_w
                self.w_EBCxHD_HD[8][64:72] = big_w
                self.w_EBCxHD_HD[9][72:80] = big_w
                self.w_EBCxHD_HD[10][80:88] = big_w
                self.w_EBCxHD_HD[11][88:96] = big_w
                self.w_EBCxHD_HD[12][96:104] = big_w
                self.w_EBCxHD_HD[13][104:112] = big_w
                self.w_EBCxHD_HD[14][112:120] = big_w
                self.w_EBCxHD_HD[15][120:128] = big_w
                self.w_EBCxHD_HD[16][128:136] = big_w
                self.w_EBCxHD_HD[17][136:144] = big_w
                self.w_EBCxHD_HD[18][144:152] = big_w
                self.w_EBCxHD_HD[19][152:160] = big_w
                self.w_EBCxHD_HD[20][160:168] = big_w
                self.w_EBCxHD_HD[21][168:176] = big_w
                self.w_EBCxHD_HD[22][176:184] = big_w
                self.w_EBCxHD_HD[23][184:192] = big_w
                self.w_EBCxHD_HD[24][192:200] = big_w
                self.w_EBCxHD_HD[25][200:208] = big_w
                self.w_EBCxHD_HD[26][208:216] = big_w
                self.w_EBCxHD_HD[27][216:224] = big_w
                self.w_EBCxHD_HD[28][224:232] = big_w
                self.w_EBCxHD_HD[29][232:240] = big_w
                self.w_EBCxHD_HD[30][240:248] = big_w
                self.w_EBCxHD_HD[31][248:256] = big_w
             
        if self.n_HD == 8:
            if self.n_EBCxHD == 64:
                self.w_EBCxHD_HD = self.weights(self.n_EBCxHD, self.n_HD, 0.003)
                self.w_EBCxHD_HD[0][0:8] = 0.5*self.HD_info.epsilon  
                self.w_EBCxHD_HD[1][8:16] = 0.5*self.HD_info.epsilon  
                self.w_EBCxHD_HD[2][16:24] = 0.5*self.HD_info.epsilon  
                self.w_EBCxHD_HD[3][24:32] = 0.5*self.HD_info.epsilon
                self.w_EBCxHD_HD[4][32:40] = 0.5*self.HD_info.epsilon
                self.w_EBCxHD_HD[5][40:48] = 0.5*self.HD_info.epsilon
                self.w_EBCxHD_HD[6][48:56] = 0.5*self.HD_info.epsilon
                self.w_EBCxHD_HD[7][56:64] = 0.5*self.HD_info.epsilon
                
        if self.n_HD == 4:
            

            if self.n_EBCxHD == 8:
                self.w_EBCxHD_HD = self.weights(self.n_EBCxHD, self.n_HD, 0.002)
                self.w_EBCxHD_HD[0][0:2] = 0.5*self.HD_info.epsilon  
                self.w_EBCxHD_HD[1][2:4] = 0.5*self.HD_info.epsilon  
                self.w_EBCxHD_HD[2][4:6] = 0.5*self.HD_info.epsilon  
                self.w_EBCxHD_HD[3][6:8] = 0.5*self.HD_info.epsilon
            
            if self.n_EBCxHD == 16:
                self.w_EBCxHD_HD = self.weights(self.n_EBCxHD, self.n_HD, 0.0035)
                self.w_EBCxHD_HD[0][0:4] = 0.5*self.HD_info.epsilon  
                self.w_EBCxHD_HD[1][4:8] = 0.5*self.HD_info.epsilon  
                self.w_EBCxHD_HD[2][8:12] = 0.5*self.HD_info.epsilon  
                self.w_EBCxHD_HD[3][12:16] = 0.5*self.HD_info.epsilon
    
            
            if self.n_EBCxHD == 32:
                self.w_EBCxHD_HD = self.weights(self.n_EBCxHD, self.n_HD, 0.45*self.HD_info.epsilon  )
                self.w_EBCxHD_HD[0][0:8] = 0.4*self.HD_info.epsilon  
                self.w_EBCxHD_HD[1][8:16] = 0.4*self.HD_info.epsilon  
                self.w_EBCxHD_HD[2][16:24] = 0.4*self.HD_info.epsilon  
                self.w_EBCxHD_HD[3][24:32] = 0.4*self.HD_info.epsilon
                
        if self.n_EBC == 4:
            if self.n_EBCxHD ==4:
                self.w_EBCxHD_EBC = self.weights(self.n_EBCxHD, self.n_EBC, 0.004)
                self.w_EBCxHD_EBC[0][0] = 0.5*self.EBC_info.epsilon  
                self.w_EBCxHD_EBC[1][1] = 0.5*self.EBC_info.epsilon  
                self.w_EBCxHD_EBC[2][2] = 0.5*self.EBC_info.epsilon  
                self.w_EBCxHD_EBC[3][3] = 0.5*self.EBC_info.epsilon
                
            
                
                
        if self.n_EBC == 2:
            if self.n_EBCxHD ==4:
                max_EBC = self.EBC_info.epsilon
                y = self.weights(self.n_EBCxHD, self.n_EBC, 0)
                y[0][0:2] = max_EBC
                y[1][2:4] = max_EBC
                self.w_EBCxHD_EBC = y
            if self.n_EBCxHD ==8:
                max_EBC = self.EBC_info.epsilon
                y = self.weights(self.n_EBCxHD, self.n_EBC, 0)
                y[0][0] = 0.01*max_EBC
                y[1][0] = 0.8*max_EBC
                
                y[1][1] = 0.01*max_EBC
                y[0][1] = 0.8*max_EBC
                
                y[1][3] = 0.01*max_EBC
                y[0][3] = 0.8*max_EBC
                
                y[0][2] = 0.01*max_EBC
                y[1][2] = 0.8*max_EBC
                
                y[0][4] = 0.01*max_EBC
                y[1][4] = 0.8*max_EBC
                
                y[1][5] = 0.01*max_EBC
                y[0][5] = 0.8*max_EBC
                
                y[0][6] = 0.01*max_EBC
                y[1][6] = 0.8*max_EBC
                
                y[1][7] = 0.01*max_EBC
                y[0][7] = 0.8*max_EBC
                
                
                self.w_EBCxHD_EBC = y
                
        """
        
        if self.n_EBC == 2:
            if self.n_EBCxHD ==4:
                max_EBC = self.EBC_info.epsilon
                y = self.weights(self.n_EBCxHD, self.n_EBC, 0)
                y[0][0:2] = max_EBC
                y[1][2:4] = max_EBC
                self.w_EBCxHD_EBC = y
            if self.n_EBCxHD ==8:
                max_EBC = self.EBC_info.epsilon
                y = self.weights(self.n_EBCxHD, self.n_EBC, 0)
                y[0][0] = 0.01*max_EBC
                y[1][0] = 0.8*max_EBC
                
                y[1][1] = 0.01*max_EBC
                y[0][1] = 0.8*max_EBC
                
                y[1][3] = 0.01*max_EBC
                y[0][3] = 0.8*max_EBC
                
                y[0][2] = 0.01*max_EBC
                y[1][2] = 0.8*max_EBC
                
                y[0][4] = 0.01*max_EBC
                y[1][4] = 0.8*max_EBC
                
                y[1][5] = 0.01*max_EBC
                y[0][5] = 0.8*max_EBC
                
                y[0][6] = 0.01*max_EBC
                y[1][6] = 0.8*max_EBC
                
                y[1][7] = 0.01*max_EBC
                y[0][7] = 0.8*max_EBC
                
                
                self.w_EBCxHD_EBC = y
                
  
        if self.n_HD == 4:
            self.w_EBCxHD_HD = self.weights(self.n_EBCxHD, self.n_HD, 0)
            self.w_EBCxHD_HD[0][0:2] = 1
            self.w_EBCxHD_HD[1][2:4] = 1
            self.w_EBCxHD_HD[2][4:6] = 1
            self.w_EBCxHD_HD[3][6:8] = 1
            self.w_EBCxHD_HD = self.w_EBCxHD_HD*self.HD_info.epsilon                
        
        
        if self.n_EBC == 4:
            max_EBC = self.EBC_info.epsilon  #epsilon clone
            scaling_EBC = max_EBC/1.5
            y = self.weights(self.n_EBCxHD, self.n_EBC, 0)
            y[0][0:4]=1
            y[1][4:8]=1
            y[2][8:12]=1
            y[3][12:16]=1
            self.w_EBCxHD_EBC = y*scaling_EBC
        
        if self.n_HD==4:
            max_HD = self.HD_info.epsilon  #epsilon clone
            scaling_HD = max_HD/1.5
            y2 = self.weights(self.n_EBCxHD,self.n_HD, 0) ##initialize weights to all be zero
            y2[0][[0,4,8,12]]=1
            y2[1][[1,5,9,13]]=1
            y2[2][[2,6,10,14]]=1
            y2[3][[3,7,11,15]]=1
            self.w_EBCxHD_HD = y2*scaling_HD
            
        if self.n_EBC == 8:
            max_EBC = self.EBC_info.epsilon  #epsilon clone
            y = self.weights(self.n_EBCxHD, self.n_HD, max_EBC*0.08)
            scaling_EBC = max_EBC/4
          #  y = self.weights(self.n_EBCxHD, self.n_EBC, 0)
            y[0][0:4]=scaling_EBC
            y[1][4:8]=scaling_EBC
            y[2][8:12]=scaling_EBC
            y[3][12:16]=scaling_EBC
            y[4][0:4]=scaling_EBC
            y[5][4:8]=scaling_EBC
            y[6][8:12]=scaling_EBC
            y[7][12:16]=scaling_EBC
            self.w_EBCxHD_EBC = y
            
        if self.n_HD==8:
            
            max_HD = self.HD_info.epsilon  #epsilon clone
            
            y2 = self.weights(self.n_EBCxHD, self.n_HD, max_EBC*0.08)
        
            scaling_HD = max_HD/4
            #y2 = self.weights(self.n_EBCxHD,self.n_HD, 0) ##initialize weights to all be zero
            y2[0][[0,4,8,12]]=scaling_HD
            y2[1][[1,5,9,13]]=scaling_HD
            y2[2][[2,6,10,14]]=scaling_HD
            y2[3][[3,7,11,15]]=scaling_HD
            y2[4][[0,4,8,12]]=scaling_HD
            y2[5][[1,5,9,13]]=scaling_HD
            y2[6][[2,6,10,14]]=scaling_HD
            y2[7][[3,7,11,15]]=scaling_HD
            self.w_EBCxHD_HD = y2
            
        if self.n_EBC == 16:
            max_EBC = self.EBC_info.epsilon  #epsilon clone
           # y = self.weights(self.n_EBCxHD, self.n_HD, max_EBC*0.08)
            y = self.weights(self.n_EBCxHD, self.n_HD, 0)
            scaling_EBC = max_EBC/4
            #  y = self.weights(self.n_EBCxHD, self.n_EBC, 0)
            y[0][0:8]=scaling_EBC
            y[1][8:16]=scaling_EBC
            y[2][16:24]=scaling_EBC
            y[3][24:32]=scaling_EBC
            y[4][32:40]=scaling_EBC
            y[5][40:48]=scaling_EBC
            y[6][48:56]=scaling_EBC
            y[7][56:64]=scaling_EBC
            y[8][0:8]=scaling_EBC
            y[9][8:16]=scaling_EBC
            y[10][16:24]=scaling_EBC
            y[11][24:32]=scaling_EBC
            y[12][32:40]=scaling_EBC
            y[13][40:48]=scaling_EBC
            y[14][48:56]=scaling_EBC
            y[15][56:64]=scaling_EBC
           
            self.w_EBCxHD_EBC = y
            
        if self.n_HD==16:
            
            max_HD = self.HD_info.epsilon  #epsilon clone
            
           # y2 = self.weights(self.n_EBCxHD, self.n_HD, max_EBC*0.08)
            y2 = self.weights(self.n_EBCxHD, self.n_HD, 0)
        
            scaling_HD = max_HD/4
            #y2 = self.weights(self.n_EBCxHD,self.n_HD, 0) ##initialize weights to all be zero
            
            y2[0][[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]]=scaling_HD
            y2[1][[1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61]]=scaling_HD
            y2[2][[2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62]]=scaling_HD
            y2[3][[3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63]]=scaling_HD
            y2[4][[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]]=scaling_HD
            y2[5][[1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61]]=scaling_HD
            y2[6][[2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62]]=scaling_HD
            y2[7][[3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63]]=scaling_HD
            y2[8][[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]]=scaling_HD
            y2[9][[1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61]]=scaling_HD
            y2[10][[2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62]]=scaling_HD
            y2[11][[3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63]]=scaling_HD
            y2[12][[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]]=scaling_HD
            y2[13][[1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61]]=scaling_HD
            y2[14][[2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62]]=scaling_HD
            y2[15][[3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63]]=scaling_HD
            self.w_EBCxHD_HD = y2
            
        """
        
        self.w_EBCxHD_EBC = self.weights(self.n_EBCxHD, self.n_EBC, self.EBC_info.rand_weights_max)

        
#############################################################################################
################################# Run Network ######################################################
####################################################################################################
   
 ### this is for transformation 2!
    def EBCxHD_activity(self,index, EBCxHD_r):
        
        #scale ego_input by distance from border
        ego_dis = self.ego_walldis[self.EBCxHD_fields[:,1].astype('int'),index]
        
        ego_dis[np.logical_or(ego_dis<0, ego_dis>self.EBCxHD_fields[:,0])] = 0 #if either -1 or greater than cuttoff for field make it zero
        
        ego_dis = ego_dis > 0
        ego_dis.astype('int')
        
        ego_input = self.EBCxHD_info.base_input * ego_dis #make uniform for now, maybe change to being exponential later
        
        #scale hd_input by cosine with prefered angle
        ##again refine based on modeling info as it comes in
        hd_input = self.HD_info.base_input * np.cos(self.EBCxHD_fields[:,2] - self.headdir_ind[index])
        hd_input[hd_input<0] = 0 #no inhibtion for opposite direction, just zero any non-positive cases
        
        ego_input = ego_input.reshape((ego_input.shape[0],1))
        hd_input = hd_input.reshape((hd_input.shape[0],1))
        #"rate" i.e. activity level of each neuron to be used for spike thresholding
        EBCxHD_r = EBCxHD_r + ego_input + hd_input
        
        spikes = np.zeros((self.EBCxHD_fields.shape[0],1), dtype = 'bool')
        
        #stuff from AK paper directly modified to scale with new dt_tau
        alpha = 0.1  #amount of activation gain you get from firing a spike
        k = 500.0; 
        Beta = 0.15; 
        dt_tau = self.dt/self.tau;
        dt_seconds = self.dt/1000.0;
        threshold = stats.uniform.rvs(0,1, size = (self.EBCxHD_fields.shape[0],1))
        
        spikes = (k*dt_seconds*(ego_input + hd_input - Beta)>threshold )*1
        
        if np.any(spikes):
            EBCxHD_r = EBCxHD_r - EBCxHD_r*dt_tau + alpha * spikes
        else:
            EBCxHD_r = EBCxHD_r - EBCxHD_r*dt_tau
            
        return spikes[:,0], EBCxHD_r


    def pdf_fun(self, x, sigma):
        # x = x - mu
        val = math.exp(-x*x/(sigma*sigma))
        return val
    
    def HD_activity(self, index, HD_r):
        #scale hd_input by cosine with prefered angle
        ##again refine based on modeling info as it comes in
         ## remeber np.cos takes in radians, not degrees!
        #hd_input = self.HD_info.base_input * np.cos((self.HD_fields[:,0] - self.headdir_ind[index])*np.pi/180) 
        """Exponential scaling for HD, does not wraparound 360!
        difs = np.array(self.HD_fields[:,0] - self.headdir_ind[index])
        hd_dif = np.array([self.pdf_fun(i, 30) for i in difs])
        hd_input = self.HD_info.base_input*hd_dif
        """

        hd_input = self.HD_info.base_input * np.cos(2*(self.HD_fields[:,0] - self.headdir_ind[index])*np.pi/180)
        hd_input[hd_input<0] = 0 #no inhibtion for opposite direction, just zero any non-positive cases
        hd_input = hd_input.reshape((hd_input.shape[0],1))
        HD_r = HD_r + hd_input
        spikes = np.zeros((self.EBCxHD_fields.shape[0],1), dtype = 'bool')
        #stuff from AK paper directly modified to scale with new dt_tau
        alpha = 0.1  #amount of activation gain you get from firing a spike
        k = 500.0; 
        Beta = 0.1; 
        dt_tau = self.dt/self.tau;
        dt_seconds = self.dt/1000.0;
        threshold = stats.uniform.rvs(0,1, size = (self.HD_fields.shape[0],1))
        #t = np.zeros((self.HD_fields.shape[0],1))
        #threshold=t+0.0075
        #threshold=t+0.009
        spikes = (k * dt_seconds * (hd_input-Beta) > threshold ) * 1

        HD_r = HD_r - HD_r*dt_tau + alpha * spikes
            
        return spikes[:,0], HD_r
    
    def EBC_activity(self, index, EBC_r):
    
        ego_dis = self.ego_walldis[self.EBC_fields[:,1].astype('int'),index]  ##this is the length = 8 vector
        ego_dis[np.logical_or(ego_dis<0, ego_dis>self.EBC_fields[:,0])] = 0 #if either -1 or greater than cuttoff for field make it zero
      #  ego_dis = ego_dis > 0 this line changes it to true/falses
        ego_dis.astype('int')
   
        ego_input = self.EBC_info.base_input * ego_dis #make uniform for now, maybe change to being exponential later
        ego_input = ego_input.reshape((ego_input.shape[0],1))
        
      
        #"rate" i.e. activity level of each neuron to be used for spike thresholding
        #print(ego_input)
        EBC_r = EBC_r + ego_input 
         #stuff from AK paper directly modified to scale with new dt_tau
        alpha = 0.1  #amount of activation gain you get from firing a spike
        k = 500.0; 
        Beta = 0.2; 
        #Beta = 0
        dt_tau = self.dt/self.tau;
        dt_seconds = self.dt/1000.0;

        threshold = stats.uniform.rvs(0,1, size = (self.EBC_fields.shape[0],1))
        #t = np.zeros((self.EBC_fields.shape[0],1))
        #threshold=t+0.02
        #threshold=t+0.03
        spikes = (k*dt_seconds*(ego_input-Beta)>threshold)*1
        EBC_r = EBC_r - EBC_r*dt_tau + alpha * spikes
        """
        if np.any(spikes):
         #   y = np.argwhere(spikes)
         #   print(ego_input[y[0][0]])
            EBC_r = EBC_r - EBC_r*dt_tau + alpha * spikes
        else:
            EBC_r = EBC_r - EBC_r*dt_tau
            """
        return spikes[:,0], EBC_r

    ##return vector with zeros except where place cells are active
    def place_activity(self, index):
        x_pos = self.x_ind[index][0]
        y_pos = self.y_ind[index][0]
        cell = self.place_grid[y_pos][x_pos]
        input_place = np.zeros(self.place_info.n_place)
        if not(cell == -1):
            for j in cell:
                input_place[j] = self.EBCxHD_info.bias_size
        return input_place
        
        
        
## call this function to have EBCxHD cells learn from EBC and HD cells for transform 1
## returns spikes, EBCxHD_r vector
    def EBCxHD_activity_learn(self,index, EBCxHD_r, EBC_r, HD_r, lat_inhib):
        input_EBC = np.sum(EBC_r*self.w_EBCxHD_EBC,0).reshape((self.n_EBCxHD,1)) 
        input_HD =  np.sum(HD_r*self.w_EBCxHD_HD,0).reshape((self.n_EBCxHD,1)) 

        #input_EBCxHD_EBCxHD = np.sum(EBCxHD_r*self.w_EBCxHD_EBCxHD,0).reshape((self.n_EBCxHD,1)) 
        #input_total = input_EBC + 10*input_HD 
    
        input_total = (input_EBC + input_HD)*3
        
        if self.EBCxHD_info.place_bias :
            place_r = self.place_activity(index)
            input_place =  np.sum(place_r*self.w_EBCxHD_place,0).reshape((self.n_EBCxHD,1)) 
            input_total = input_place + input_total

        ## toggle to see if want to include place cell bias
        ### this is a very simple grid-like implementation 
        ### it did not work very well
        """
        if self.EBCxHD_info.place_bias:
           
            x_pos = self.x_ind[index]
            y_pos = self.y_ind[index]

            ## hard-coded for 64 cells
            x_bin = round(x_pos[0] / 41)
            y_bin = round(y_pos[0] / 41)
            cell = x_bin * 8 + y_bin
            self.list_loc.append(cell)  ##list to keep track of which cell is getting the extra bias
            input_total[cell] += self.EBCxHD_info.bias_size
        
            
        if self.EBCxHD_info.place_bias:
            x = self.x_ind[index]
            y = self.y_ind[index]
            xbin = round(x[0]/75)
            ybin = round(y[0]/75)
            box = xbin * 4 + ybin
            boxes = self.place_fields[:,3]
            searchval = box
            place_indices = np.where(boxes == searchval)[0]
            box_list.append(box)
            for i in range(len(place_indices)):
                input_total[i] += self.EBCxHD_info.bias_size
                input_total[i + 32]
            
       
        if self.EBCxHD_info.place_bias:
            x = self.x_ind[index]
            y = self.y_ind[index]
            xbin = round(x[0]/100)
            ybin = round(y[0]/100)
            box = xbin * 3 + ybin
            print(box)
            input_total[7*box:7*box+7] += self.EBCxHD_info.bias_size
            
            """
        if lat_inhib < 1:
            ## update from EBC and HD input
            EBCxHD_r = EBCxHD_r + input_total
        else:
       #     EBCxHD_active = [arr[index] for arr in EBCxHD_spikes]
            
        #    HD_active = HD_spikes[index]
            # "need to double check math for lateral inhibition for transformation 1"
         #   mask = EBCxHD_r > 0 #don't do inhibition for cells that aren't active
          #  pos_r = mask*EBCxHD_r
         #  EBCxHD_r = EBCxHD_r + input_total - self.inhib_weight*mask*(np.sum(EBCxHD_r)-EBCxHD_r) #get n_ABC x 1 vector with the activity of the same cell removed
          #  EBCxHD_r = EBCxHD_r + input_total - self.EBCxHD_info.inhib_weight * mask * (np.sum(pos_r) - EBCxHD_r)
            
            #Ron's inhibition function
            
            mask =  (EBCxHD_r > 0)*1
            remove_self = np.ones((self.n_EBCxHD,self.n_EBCxHD)) - np.eye(self.n_EBCxHD,self.n_EBCxHD)
            mask = mask * remove_self        
            input_sub = self.EBCxHD_info.inhib_weight * np.sum(mask*EBCxHD_r,0, keepdims = True).T
            EBCxHD_r = EBCxHD_r + input_total - input_sub 
            EBCxHD_r[EBCxHD_r<0]=0 
            
            EBCxHD_r = EBCxHD_r + input_total
            
            """
            bools = EBCxHD_r>0.75
            EBCxHD_r = EBCxHD_r + input_total
            
            if np.sum(bools)>0:
                
                inhib_r = np.zeros_like(EBCxHD_r)
                #first = next(x[0] for x in enumerate(EBCxHD_r) if x[1] > 0.4) ## returns index of first element greater than threshold
                first = np.argmax(EBCxHD_r)
                inhib_r = inhib_r - 0.000005
                inhib_r[first] = 0
                EBCxHD_r = EBCxHD_r + inhib_r
                EBCxHD_r[EBCxHD_r<0]=0.0001 
                
                ## check for highest input
                ## everything else gets halved
                ## highest input stays constant

                first = np.argmax(EBCxHD_r.flatten())
                val = EBCxHD_r[first]
                EBCxHD_r = 0.5*EBCxHD_r
                EBCxHD_r[first] = val
                EBCxHD_r[EBCxHD_r<0]=0.00001 
                """
            ## trying new inhibition mechanism
            
            #stuff from AK paper directly modified to scale with new dt_tau
        alpha = 0.1
        k = 500.0; 
        Beta = 0.3; #2019-09-04 o.g. paper had this at zero but 0.075 seems to get most realistic peak firing rates so leaving that for now
        dt_tau = self.dt/self.tau;
        dt_seconds = self.dt/1000.0;
        threshold = stats.uniform.rvs(0,1, size = (self.n_EBCxHD,1))
        #t = np.zeros((self.EBCxHD_fields.shape[0],1))
       # threshold = t + 0.21
        spikes = (k*dt_seconds*(input_total-Beta)>threshold )*1
        #EBCxHD_r = EBCxHD_r - EBCxHD_r*dt_tau + alpha * spikes
        EBCxHD_r = EBCxHD_r - EBCxHD_r*dt_tau + alpha * spikes
        """
        if np.any(spikes):
            EBCxHD_r = EBCxHD_r - EBCxHD_r*dt_tau + alpha * spikes
        else:
            EBCxHD_r = EBCxHD_r - EBCxHD_r*dt_tau
            
            
            ## trying putting inhibition after updating rate
        if lat_inhib == 1:
            mask =  (ABC_r > 0)*1 
   
            ##check if there were any spikes
             check  = spikes > 0
             
             ## if there were spikes, then for each spikes, inhibit all other cells besides self
             if np.any(check):
                 ## for each spike
                 for i in range(len(spikes)):
                    if spikes[i] == 1:
                        basic = np.ones(len(spikes))
                        basic[i] = 0
                        basic = basic.reshape((self.n_EBCxHD,1))
                        EBCxHD_r = EBCxHD_r - self.EBCxHD_info.inhib_weight * basic
                        x = EBCxHD_r
                        EBCxHD_r = np.where(x>0, x, 0)
                        #EBCxHD_r = [0 if num < 0 else num for num in EBCxHD_r]
                """
        
        if self.EBCxHD_info.place_bias:      
            return spikes[:,0], EBCxHD_r
        else:
            return spikes[:,0], EBCxHD_r
        

    def ABC_activity(self,index,ABC_r, EBCxHD_r,lat_inhib): #come back to decision to only allowing communication when spike occured; not done in other models (reflecting time it takes for synapse to fully clear)
    
        #also debate on using lateral inhibition here
        #not sure really need index but leaving it in for now
        if lat_inhib < 1:
            ABC_r = ABC_r + np.sum(EBCxHD_r*self.w_ABC_EBCxHD,0).reshape((self.n_ABC,1)) #axis confirmed correct
        else:
            # old - mask = ABC_r > 0 #don't do inhibition for cells that aren't active
            #old - ABC_r = ABC_r + np.sum(EBCxHD_r*self.w_ABC_EBCxHD,0).reshape((self.n_ABC,1)) - self.ABC_info.inhib_weight*mask*(np.sum(ABC_r)-ABC_r) #get n_ABC x 1 vector with the activity of the same cell removed
            n_ABC = self.n_ABC
            inhib_weight = self.ABC_info.inhib_weight
            w_ABC_EBCxHD = self.w_ABC_EBCxHD
            
            mask =  (ABC_r > 0)*1
            remove_self = np.ones((n_ABC,n_ABC)) - np.eye(n_ABC,n_ABC)
            mask = mask *remove_self
        
            input_total = np.sum(EBCxHD_r*w_ABC_EBCxHD,0).reshape((n_ABC,1)) - inhib_weight* np.sum(mask*ABC_r,0, keepdims = True).T#(np.sum(mask*ABC_r)-mask*ABC_r) #get n_ABC x 1 vector with the activity of the same cell removed
            input_total = 30*np.sum(EBCxHD_r*w_ABC_EBCxHD,0).reshape((n_ABC,1))
            ABC_r[ABC_r<0]=0 
            ABC_r = ABC_r + input_total
        spikes = np.zeros((self.n_ABC,1), dtype = 'bool')
        #stuff from AK paper directly modified to scale with new dt_tau
        alpha = 0.1
        k = 500.0 
        Beta = 0.3 #2019-09-04 o.g. paper had this at zero but 0.075 seems to get most realistic peak firing rates so leaving that for now
        dt_tau = self.dt/self.tau
        dt_seconds = self.dt/1000.0
        threshold = stats.uniform.rvs(0,1, size = (self.n_ABC,1))
        spikes = (k*dt_seconds*(input_total-Beta)>threshold )*1
        
        ABC_r = ABC_r - 2*ABC_r*dt_tau + alpha * spikes
       
        """
        if np.any(spikes):
            ABC_r = ABC_r - ABC_r*dt_tau + alpha * spikes
            
        else:
            ABC_r = ABC_r - ABC_r*dt_tau
        
        """
        return spikes[:,0], ABC_r


    def update_weights(self, weights,aj,ai, epsilon_ij, learn_rate): #triple check this learning rule!!!!!!
        #ai is activty FROM (i.e. presynaptic)
        #aj is activty TO (i.e. postsynaptic)
        
        #Parameters
        lambda_lr = learn_rate #learning rate
        #this is tuneable as well, need to think how to handle when we are updating other weights, probably just add toggle

   #    mask = weights > 0 #only include weights with non-negative weights in final sum, so have to include this check    
   #     weights = weights + lambda_lr * aj[:,0]*(ai*(epsilon_ij - weights) - weights*mask*(np.sum(ai)-ai))
        
        mask = (weights != 0)*1 #only include weights with non-zero weights in final sum, so have to include this check
         #everything should work now
        weights = weights + lambda_lr *aj[:,0]*((epsilon_ij-weights)*ai - weights*(np.matmul(ai.T,mask)-np.multiply(mask,ai)))
        return weights

        ## new version of update weights rule
        
        ## only updates weights to cell with maximum activity 
    def update_weights_v2(self, weights,aj,ai, epsilon_ij, learn_rate): #triple check this learning rule!!!!!!
        #ai is activty FROM (i.e. presynaptic)
        #aj is activty TO (i.e. postsynaptic)
        
        lambda_lr = learn_rate #learning rate
        max_EBCxHD = np.argmax(aj)  ##index of EBCxHD cell with  rate
        wij_vec = weights[:,max_EBCxHD] ## column of weights to update
        ai_flat = ai.flatten()
        mask = wij_vec != 0 ##only sum over units with weights not equal to 0
        
        ##initialize array to look like ai, add in the sum of all rates then subtract each individual one        
        full_sum_incoming = np.sum(mask * ai_flat) 
        sum_subtract = np.zeros_like(ai_flat) + full_sum_incoming - ai_flat 
        sub = wij_vec * sum_subtract
        
        add = (epsilon_ij - wij_vec) * ai_flat
        w_new = wij_vec + lambda_lr * aj[max_EBCxHD][0] * (add - sub)
        weights[:,max_EBCxHD] = w_new

        """
        max_EBCxHD = np.argmax(aj)
        ## using old rule to do new
        mask = (weights != 0)*1
        weights_update = weights + lambda_lr *aj[:,0]*((epsilon_ij-weights)*ai - weights*(np.matmul(ai.T,mask)-np.multiply(mask,ai)))
         ## extract cell with max activation
        weights[:,max_EBCxHD] = weights_update[:,max_EBCxHD] ## copy over only max new row
        """
        return weights


    ## updates all weights for aj above certain threshold
    def update_weights_v3(self, weights,aj,ai, epsilon_ij, learn_rate): #triple check this learning rule!!!!!!
        #ai is activty FROM (i.e. presynaptic)
        #aj is activty TO (i.e. postsynaptic)
        
        lambda_lr = learn_rate #learning rate
        mean = np.mean(aj)
        std = np.std(aj)
        to_update = aj > (mean + 0.25*std)
        """
        #max_EBCxHD = np.argmax(aj)  ##index of EBCxHD cell with  rate
        for i in range(len((to_update))):
            if (to_update[i][0]):
                max_EBCxHD = i
                wij_vec = weights[:,max_EBCxHD] ## column of weights to update
                ai_flat = ai.flatten()
                mask = wij_vec>0 ##only sum over units with weights > 0
                
                ##initialize array to look like ai, add in the sum of all rates then subtract each individual one        
                full_sum_incoming = np.sum(mask * ai_flat) 
                sum_subtract = np.zeros_like(ai_flat) + full_sum_incoming - ai_flat 
                sub = wij_vec * sum_subtract
                
                add = (epsilon_ij - wij_vec) * ai_flat
                w_new = wij_vec + lambda_lr * aj[max_EBCxHD][0] * (add - sub)
                weights[:,max_EBCxHD] = w_new
                
                """
        mask = (weights != 0)*1
        weights_update = weights + lambda_lr *aj[:,0]*((epsilon_ij-weights)*ai - weights*(np.matmul(ai.T,mask)-np.multiply(mask,ai)))
       
        for i in range(len((to_update))):
            if (to_update[i][0]):
                weights[:,i] = weights_update[:,i] ## copy over only max new row
                
        return weights
    
    
    ## updates all weights for aj above certain threshold
    def update_weights_v4(self, weights,aj,ai, epsilon_ij, learn_rate): #triple check this learning rule!!!!!!
        #ai is activty FROM (i.e. presynaptic)
        #aj is activty TO (i.e. postsynaptic)
        
        lambda_lr = learn_rate #learning rate
        mean = np.mean(aj)
        std = np.std(aj)
        to_update = aj > (mean + std)
        """
        #max_EBCxHD = np.argmax(aj)  ##index of EBCxHD cell with  rate
        for i in range(len((to_update))):
            if (to_update[i][0]):
                max_EBCxHD = i
                wij_vec = weights[:,max_EBCxHD] ## column of weights to update
                ai_flat = ai.flatten()
                mask = wij_vec>0 ##only sum over units with weights > 0
                
                ##initialize array to look like ai, add in the sum of all rates then subtract each individual one        
                full_sum_incoming = np.sum(mask * ai_flat) 
                sum_subtract = np.zeros_like(ai_flat) + full_sum_incoming - ai_flat 
                sub = wij_vec * sum_subtract
                
                add = (epsilon_ij - wij_vec) * ai_flat
                w_new = wij_vec + lambda_lr * aj[max_EBCxHD][0] * (add - sub)
                weights[:,max_EBCxHD] = w_new
                
                """
        mask = (weights != 0)*1
        weights_update = weights + lambda_lr *aj[:,0]*((epsilon_ij-weights)*ai - weights*(np.matmul(ai.T,mask)-np.multiply(mask,ai)))
       
        for i in range(len((to_update))):
            if (to_update[i][0]):
                weights[:,i] = weights_update[:,i] ## copy over only max new row
                
        return weights
     
    def run_network(self, num_transform):
        
        rinit = 1e-3 #taking intial activation value from grid code
        
        """Transformation 1 """
        if (num_transform == 1):    
            ### initialize all spike trains to zero
            HD_spikes = np.zeros((self.HD_fields.shape[0],self.time_ind))  
            EBC_spikes = np.zeros((self.EBC_fields.shape[0],self.time_ind))
            EBCxHD_spikes = np.zeros((self.n_EBCxHD,self.time_ind))
            ABC_spikes = [] ## won't be needing these really
            ABC_r = []
            
            ## intialize initial firing rates to rinit
            HD_r = np.ones((self.HD_fields.shape[0],1))*rinit
            EBC_r = np.ones((self.EBC_fields.shape[0],1))*rinit
            EBCxHD_r = np.ones((self.n_EBCxHD,1))*rinit
            lat_inhib = 1  ## 1 means there IS lateral inhib, 0 means NO lateral inhib

            rates = np.zeros([self.n_EBCxHD, self.time_ind])
           
            ## Begin iterating through time          
            print("Start")

        
            #for index in range(1):   
          #  for index in range(9000):   
            for index in range(round(self.time_ind)):   

                if np.mod(index,100000)==0:
                    print(index)  ## prints out where you are for time updates
              
                [HD_spikes[:,index],HD_r] = self.HD_activity(index,HD_r)  ##update activity of HD spikes
                [EBC_spikes[:,index], EBC_r] = self.EBC_activity(index, EBC_r) ## update activity of EBC spikes
                    
                ##update activity of conjuctive cells based on HD and EBC inputs
                [EBCxHD_spikes[:,index], EBCxHD_r] = self.EBCxHD_activity_learn(index, EBCxHD_r, EBC_r, HD_r, lat_inhib)  
                rates[:,index] = EBCxHD_r.flatten()

                    
                ## update weights separately
                if self.learn:
                    self.w_EBCxHD_HD = self.update_weights(self.w_EBCxHD_HD, EBCxHD_r, HD_r, self.HD_info.epsilon, self.HD_info.learn_rate)
                    self.w_EBCxHD_EBC = self.update_weights_v2(self.w_EBCxHD_EBC, EBCxHD_r, EBC_r, self.EBC_info.epsilon, self.EBC_info.learn_rate) 
                    
                    ##store current weight matrices
                #    self.all_w_EBCxHD_HD[index] = self.w_EBCxHD_HD
                 #   self.all_w_EBCxHD_EBC[index] =  self.w_EBCxHD_EBC
        

            self.rates = rates
            self.HD_spikes, self.EBC_spikes, self.EBCxHD_spikes, self.EBCxHD_r, self.HD_r, self.EBC_r = HD_spikes, EBC_spikes, EBCxHD_spikes, EBCxHD_r, HD_r, EBC_r
                
        """Transformation 2 """
        if (num_transform == 2):
            
            #Initialize spikes/rate vectors       
            EBCxHD_spikes = np.zeros((self.EBCxHD_fields.shape[0],self.time_ind))
            EBCxHD_r = np.ones((self.EBCxHD_fields.shape[0],1))*rinit
            ABC_spikes = np.zeros((self.n_ABC,self.time_ind))
            ABC_r = np.ones((self.n_ABC,1))*rinit
            HD_spikes = []
            EBC_spikes = []
            lat_inhib = 1 #toggle whether allow constant lateral inhibition between ABCs
        
            ## Begin iterating through time
        #    for index in range(0,self.time_ind): #Note need to add _r to everything we haven't so far
            for index in range(0,10):
          #  for index in range(0,10): #Note need to add _r to everything we haven't so far
                if np.mod(index,10000)==0:
                                print(index)  ## prints out where you are for time updates
        
                [EBCxHD_spikes[:,index], EBCxHD_r] = self.EBCxHD_activity(index, EBCxHD_r)  ## updates EBCxHD activity
                [ABC_spikes[:,index], ABC_r ]= self.ABC_activity(index,ABC_r, EBCxHD_r,lat_inhib)     ## update ABC activity 
                self.w_ABC_EBCxHD = self.update_weights(self.w_ABC_EBCxHD,ABC_r,EBCxHD_r, self.EBCxHD_info.epsilon)  ##  update weights between ABCs and EBCxHD's
                self.EBCxHD_spikes, self.ABC_spikes, self.EBCxHD_r, self.ABC_r = EBCxHD_spikes, ABC_spikes, EBCxHD_r, ABC_r
                
        """ Transfomation 3 (both 1 and 2)"""
        if (num_transform == 3): 
                    
            ##initialize all spike trains
            HD_spikes = np.zeros((self.HD_fields.shape[0],self.time_ind))
            EBC_spikes = np.zeros((self.EBC_fields.shape[0],self.time_ind))
            EBCxHD_spikes = np.zeros((self.EBCxHD_fields.shape[0],self.time_ind))
            ABC_spikes = np.zeros((self.n_ABC,self.time_ind))
           
            ##initialize firing rates
            EBCxHD_r = np.ones((self.EBCxHD_fields.shape[0],1))*rinit
            HD_r = np.ones((self.HD_fields.shape[0],1))*rinit
            EBC_r = np.ones((self.EBC_fields.shape[0],1))*rinit
            EBCxHD_r = np.ones((self.n_EBCxHD,1))*rinit
            ABC_r = np.ones((self.n_ABC,1))*rinit
        
            lat_inhib = 1
            
            self.rates_ABC = np.zeros([self.n_ABC, self.time_ind])
            self.rates_EBCxHD = np.zeros([self.n_EBCxHD, self.time_ind])
            self.rates_EBC = np.zeros([self.n_EBC, self.time_ind])
            self.rates_HD = np.zeros([self.n_HD, self.time_ind])
            self.rates_place = np.zeros([self.n_place, self.time_ind])
            
            #for index in range(0,10000): 
            for index in range(0,self.time_ind): 
            #for index in range(0,3):   #
                if np.mod(index,100000)==0:
                    print(index)
                [HD_spikes[:,index],HD_r] = self.HD_activity(index,HD_r)  ##update activity of HD spikes
                [EBC_spikes[:,index], EBC_r] = self.EBC_activity(index, EBC_r) ## update activity of EBC spikes
                    
                ##update activity of conjuctive cells based on HD and EBC inputs
                [EBCxHD_spikes[:,index], EBCxHD_r] = self.EBCxHD_activity_learn(index, EBCxHD_r, EBC_r, HD_r, lat_inhib)  
               
                # update activity of ABCs based on learned EBCxHD activity
                [ABC_spikes[:,index], ABC_r] = self.ABC_activity(index, ABC_r, EBCxHD_r,lat_inhib)
                
                ## store rates of each cell
                self.rates_ABC[:,index] = ABC_r.flatten()
                self.rates_EBCxHD[:,index] = EBCxHD_r.flatten()
                self.rates_EBC[:,index] = EBC_r.flatten()
                self.rates_HD[:,index] = HD_r.flatten()
                
                if self.learn:   
                       ##  update_weights(self, weights,aj,ai, epsilon_ij, learn_rate)
                    self.w_ABC_EBCxHD = self.update_weights_v3(self.w_ABC_EBCxHD,ABC_r,EBCxHD_r, self.EBCxHD_info.epsilon, self.EBCxHD_info.learn_rate)     
                    self.w_EBCxHD_HD = self.update_weights_v3(self.w_EBCxHD_HD, EBCxHD_r, HD_r, self.HD_info.epsilon, self.HD_info.learn_rate) 
                    self.w_EBCxHD_EBC= self.update_weights_v3(self.w_EBCxHD_EBC, EBCxHD_r, EBC_r, self.EBC_info.epsilon, self.EBC_info.learn_rate) 
                    

            self.EBCxHD_spikes, self.ABC_spikes, self.EBCxHD_r, self.ABC_r = EBCxHD_spikes, ABC_spikes, EBCxHD_r, ABC_r
            self.HD_spikes, self.EBC_spikes, self.EBCxHD_spikes, self.EBCxHD_r, self.HD_r, self.EBC_r = HD_spikes, EBC_spikes, EBCxHD_spikes, EBCxHD_r, HD_r, EBC_r
                 
      #  return HD_spikes,EBC_spikes, EBCxHD_spikes, ABC_spikes, self.w_ABC_EBCxHD, self.w_EBCxHD_HD, ABC_r, EBCxHD_r #leaving r's only for debug
    
