#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 14:39:24 2021

@author: sabinechavin
"""

import matplotlib.pyplot as plt
import numpy as np
from astropy.stats import rayleightest
from astropy import units as u
import os
import math


"""Summary of plotting functions in this module
@params
net - network object from clean_Network class
save - boolean parameter to save or not
path - string path to save plot along
so far this is only setup for transform 1 mainly

## polar plots
polar_HD_input_fields(net, save, path) - plots the firing field of HD cells on a polar plot
polar_EBC_input_fields(net, save, path) - plots firing field of EBC cells on a polar plot
polar_EBCxHD_HD_firing(net, save, path) - plots HD tuning curve of conjunctive EBCxHD cells on  polar plot
polar_EBCxHD_EB_firing(net, save, path) - plots EB tuning curve of conjunctive EBCxHD cells on polar plot

## normal histograms
summary_hist_EBC_firing(net, save, path) - plots all tuning curves for EBC cells as histogram
summary_hist_HD_firing(net, save, path)

summary_hist_EBCxHD_EB_tuning - plots EB tuning curves for EBCxHD cells
summary_hist_EBCxHD_HD_tuning(net,save,path)

## visualizing weights
EBCxHD_HD_weights(net, save, path) - plots weights between HD cells and EBCxHD cells
EBCxHD_EBC_weights(net, save, path) - plots weights between EBC cells and EBCxHD cells

# calculate rayleigh vector length
rayleigh(spikes, headdir_ind)
"""


#########
##Polar Plots
########

## need to change indexing of subplots to be function of number of cells, not as hardcoded right now
## plots firing pattern of HD input cells as a polar plot
def polar_HD_input_fields(net,save,path):
    
    
    fig,ax = plt.subplots(5,5,subplot_kw={'projection': 'polar'})
    plt.subplots_adjust(hspace=0.5)
    
    for x in range(5):
        for y in range(5):
            ind = 5*x + y
            spikes = net.HD_spikes[ind] ## HD spike train for the first neuron
            angles = net.headdir_ind[spikes>0]
            hist, bin_edges = np.histogram(angles,360, range=(-180,180))
            theta_rad =  bin_edges[:-1] * np.pi/180
            ax[x,y].plot(theta_rad, hist)
            ax[x,y].set_title("Cell #" + str(ind))
            
            ## calculate Rayleigh score
            score = rayleigh(spikes,net.headdir_ind)
            ax[x,y].set_title(str(round(score,3)))
           # ax[x,y].set_title("Rayleigh Score =" + str(round(score,3)))
    
    plt.suptitle("Polar Plots of HD Input Firing Patterns") 
    
 ## plots firing pattern of EBC input cells as a polar plot
def polar_EBC_input_fields(net,save,path):
    fig,ax = plt.subplots(2,2,subplot_kw={'projection': 'polar'})
    plt.subplots_adjust(hspace=0.5)
    num_angles = len(net.ego_walldis)
    time = net.time_ind
    angles = [0,1,2,3,4,5,6,7,0]
    theta_rad = [num * 45 * np.pi/180 for num in angles]
    
    for x in range(2):
        for y in range(2):
            ind = 2*x + y
            spikes = net.EBC_spikes[ind]
            result = []
            ## loop through every time step and check if there is a spike
            for t in range(time):
                if spikes[t] == 1:
                    
                    ## for every spike, loop through every angle and see if there is a wall present
                    for i in range(num_angles):
                        if net.ego_walldis[i][t] > 0:
                            result = np.append(result,[i])
                            
            hist, bin_edges = np.histogram(result,8, range=(0,8), density=False)
            hist = np.append(hist,[hist[0]])  ##appending last data point closes circle in plots
            ax[x,y].plot(theta_rad, hist)
            
            ## calculate rayleigh score
            results_deg = [num*45 for num in result]*u.deg
            score = float(rayleightest(results_deg))
            ax[x,y].set_title("Cell #" + str(ind))
            
    plt.suptitle("Polar Plots of EBC Input Firing Patterns ") 
    
  ## Plot polar plots for conjunctive EBCxHD cells head direction firing patterns
def polar_EBCxHD_HD_firing(net,save,path):
    num_conj = net.n_EBCxHD
    sqr = int(np.round(np.sqrt(num_conj)))
    
    fig,ax = plt.subplots(sqr,sqr,subplot_kw={'projection': 'polar'},figsize=(15,15))
    plt.subplots_adjust(hspace=0.5)
    for x in range(sqr):
        for y in range(sqr):
            ind = sqr*x + y
            if ind<num_conj:
                spikes = net.EBCxHD_spikes[ind] ## HD spike train for the first neuron
                angles = net.headdir_ind[spikes>0]
                hist, bin_edges = np.histogram(angles,360, range=(0,360), density=True)
                theta_rad =  np.linspace(0,2*np.pi,360)
                ax[y,x].plot(theta_rad, hist,'r')
            ax[x,y].set_title("Cell #" + str(ind))
            
            #calculate Rayleigh score
            score = rayleigh(spikes,net.headdir_ind)
            ax[x,y].set_title(str(round(score,3)))
            

            
    
    plt.suptitle("Polar Plots of Conjunctive EBCxHD Cell HD Firing Patterns ") 

## Plot polar plots for conjunctive EBCxHD cells egocentric boundaries firing patterns
def polar_EBCxHD_EB_firing(net,save,path):
    ## Plot polar plots for EBCxHD EBC firing patterns
    fig,ax = plt.subplots(4,4,subplot_kw={'projection': 'polar'})
    plt.subplots_adjust(hspace=0.5)
    num_angles = len(net.ego_walldis)
    time = net.time_ind
    angles = [0,1,2,3,4,5,6,7,0]
    theta_rad = [num * 45 * np.pi/180 for num in angles]
    
    for x in range(4):
        for y in range(4):
            ind = 4*x + y
            spikes = net.EBCxHD_spikes[ind]
            result = []
            ## loop through every time step and check if there is a spike
            for t in range(time):
                if spikes[t] == 1:
    
                    ## for every spike, loop through every angle and see if there is a wall present
                    for i in range(num_angles):
                        if net.ego_walldis[i][t] > 0:
                            result = np.append(result,[i])
            print(len(result))
            hist, bin_edges = np.histogram(result,8, range=(0,8), density=True)
            hist = np.append(hist,[hist[0]])  ##appending last data point closes circle in plots
            ax[x,y].plot(theta_rad, hist)
            
            ## calculate rayleigh score
            """
            results_deg = [num*45 for num in result]*u.deg
            score = float(rayleightest(results_deg))
            ax[x,y].set_title("Cell #" + str(ind) + ", Rayleigh Score =" + str(round(score,3)))
            """
            
    plt.suptitle("Polar Plots of EBCxHD EBC Firing Patterns v1 full time") 

########################
####Summary histograms
########################
def summary_hist_EBC_firing(net, save, path):
    
    fig, axs = plt.subplots(2, 2,sharex = True, sharey = True)
    EBC_spikes = net.EBC_spikes
    for x in range(2):
        for y in range(2):
            ind = x * 2 + y
            result = []
            int_arr = EBC_spikes
            spikes = int_arr[ind]
            for t in range(net.time_ind):
                if spikes[t] == 1:
                    for i in range(len(net.ego_walldis)):
                        if net.ego_walldis[i][t] > 0:
                            result = np.append(result,[i])
                        
            axs[x,y].hist(result, bins =[0,1,2,3,4,5,6,7,8], ec='black', align = 'left')
    plt.suptitle("Input Egocentric boundaries histogram plots")
    if save == True:
        if not(os.path.isdir(path)):
            os.makedirs(path)
        plt.savefig(path + "/EBC_summary_hists.png")
 
def summary_hist_HD_firing(net, save, path):      
        ## visualize all head direction plots on shared axes
    fig, axs = plt.subplots(2, 2,sharex = True, sharey = True)
    ## plot histograms of ego wall dis  
    int_arr = net.HD_spikes
    for i in range(2):
        for j in range(2):
            ind = i * 2 + j
            spikes = int_arr[ind]
            x = net.headdir_ind[spikes>0]
            axs[i,j].hist(x)
            
    plt.suptitle("Head direction histogram plots")
    if save == True:
        if not(os.path.isdir(path)):
            os.makedirs(path)
        plt.savefig(path + "/HD_summary_hists.png")
        
def summary_hist_EBCxHD_EB_tuning(net,save,path):
    fig, axs = plt.subplots(4, 4,sharex = True, sharey = True)
    for x in range(4):
        for y in range(4):
            ind = x * 4 + y
            int_arr = net.EBCxHD_spikes
            result = []
            spikes = int_arr[ind]
            for t in range(net.time_ind):
                if spikes[t] == 1:
                    for i in range(len(net.ego_walldis)):
                        if net.ego_walldis[i][t] > 0:
                            result = np.append(result,[i])
                       
            axs[x,y].hist(result, bins =[0,1,2,3,4,5,6,7,8], ec='black', align = 'left')
    plt.suptitle("Egocentric boundaries histogram plots " )
    if save:
        if not(os.path.isdir(path)):
            os.makedirs(path)
            plt.savefig(path + "/summary_hists_EBCxHD_EB_tuning.png")

def summary_hist_EBCxHD_HD_tuning(net,save,path):
    ## visualize all head direction plots on shared axes
    fig, axs = plt.subplots(4, 4,sharex = True, sharey = True)
    ## plot histograms of ego wall dis   
    int_arr = net.EBCxHD_spikes 
    for i in range(4):
        for j in range(4):
            ind = i * 4 + j
            spikes = int_arr[ind]
            x = net.headdir_ind[spikes>0]
            axs[i,j].hist(x)
    plt.suptitle("EBCxHD Head direction histogram plots")
    if not(os.path.isdir(path)):
        os.makedirs(path)
        plt.savefig(path + "/summary_hists_EBCxHD_EB_tuning.png")
    plt.suptitle("Head direction  histogram plots")

########################
####Weight Plotting Functions
########################

def EBCxHD_HD_weights_final(net, save, path):
    minimum = min(np.amin(net.w_EBCxHD_HD), np.amin(net.init_w_EBCxHD_HD))
    maximum = max(np.amax(net.w_EBCxHD_HD),np.amax(net.init_w_EBCxHD_HD))
    
    plt.figure(figsize=(20,10))
    w_EBCxHD_HD = net.w_EBCxHD_HD
    plt.imshow(w_EBCxHD_HD, vmin=minimum, vmax=maximum )

    if net.n_HD == 4:
        plt.yticks([0,1,2,3], labels = ["North", "West", "South", "East"])
    if net.n_HD == 8:
        plt.yticks([0,1,2,3,4,5,6,7], labels = ["North", "West", "South", "East","North", "West", "South", "East"])
    plt.colorbar()
    plt.title("Final w_EBCxHD_HD")
    if save:
        if not(os.path.isdir(path)):
            os.makedirs(path)
        plt.savefig(path + "/headdir_weights_final.png")
    
def EBCxHD_HD_weights_init(net, save, path):
    minimum = min(np.amin(net.w_EBCxHD_HD), np.amin(net.init_w_EBCxHD_HD))
    maximum = max(np.amax(net.w_EBCxHD_HD),np.amax(net.init_w_EBCxHD_HD))
    
    plt.figure(figsize=(20,10))
    w_EBCxHD_HD = net.init_w_EBCxHD_HD
    plt.imshow(w_EBCxHD_HD, vmin=minimum, vmax=maximum )
    if net.n_HD == 4:
        plt.yticks([0,1,2,3], labels = ["North", "West", "South", "East"])
    if net.n_HD == 8:
        plt.yticks([0,1,2,3,4,5,6,7], labels = ["North", "West", "South", "East","North", "West", "South", "East"])
    plt.colorbar()
    plt.title("Initial w_EBCxHD_HD")
    if save:
        if not(os.path.isdir(path)):
            os.makedirs(path)
        plt.savefig(path + "/headdir_weights_init.png")
        
def EBCxHD_EBC_weights_final(net, save, path):   
    minimum = min(np.amin(net.w_EBCxHD_EBC), np.amin(net.init_w_EBCxHD_EBC))
    maximum = max(np.amax(net.w_EBCxHD_EBC),np.amax(net.init_w_EBCxHD_EBC))
    
    plt.figure(figsize=(20,10))
    w_EBCxHD_EBC = net.w_EBCxHD_EBC
    plt.imshow(w_EBCxHD_EBC, vmin=minimum, vmax=maximum)
    if net.n_EBC == 4:
        plt.yticks([0,1,2,3], labels = ["Backwards", "Forwards", "Right", "Left"])
    if net.n_EBC == 8:
        plt.yticks([0,1,2,3,4, 5,6,7], labels = ["Backwards", "Forwards", "Right", "Left","Backwards", "Forwards", "Right", "Left"])
    plt.colorbar()
    plt.title("Final w_EBCxHD_EBC")
    if save:
        if not(os.path.isdir(path)):
            os.makedirs(path)
        plt.savefig(path + "/ego_wall_dis_weights_final.png")
        
def EBCxHD_EBC_weights_init(net, save, path):    
    minimum = min(np.amin(net.w_EBCxHD_EBC), np.amin(net.init_w_EBCxHD_EBC))
    maximum = max(np.amax(net.w_EBCxHD_EBC),np.amax(net.init_w_EBCxHD_EBC))
    
    plt.figure(figsize=(20,10))
    w_EBCxHD_EBC = net.init_w_EBCxHD_EBC
    plt.imshow(w_EBCxHD_EBC, vmin=minimum, vmax=maximum)
    if net.n_EBC == 4:
        plt.yticks([0,1,2,3], labels = ["Backwards", "Forwards", "Right", "Left"])
    if net.n_EBC == 8:
        plt.yticks([0,1,2,3,4,5,6,7], labels = ["Backwards", "Forwards", "Right", "Left","Backwards", "Forwards", "Right", "Left"])
    plt.colorbar()
    plt.title("Initial w_EBCxHD_EBC")
    if save:
        if not(os.path.isdir(path)):
            os.makedirs(path)
        plt.savefig(path + "/ego_wall_dis_weights_init.png")
      

def ABC_EBCxHD_weights_final(net, save, path):   
    plt.imshow(net.w_ABC_EBCxHD)    
    
######
##visualize how much the weights are changing by at each time step
def weights_traj(net,save,path):
    x =net.all_w_EBCxHD_HD
    diff = np.zeros(len(x-1))
    y = abs(x[1:len(x)] - x[0:len(x)-1])
    diff = [np.sum(y[i]) for i in range(len(y))]
    plt.figure()
    plt.scatter(np.arange(0,len(diff)),diff)
    plt.xlabel("Time Step")
    plt.ylabel("Absolute Change in Weight Matrix")
    if save:
        if not(os.path.isdir(path)):
            os.makedirs(path)
        plt.savefig(path + "/change_weights.png")


#### 
#calculating Rayleigh vector for given set of angles/headdir data
####
def rayleigh(spikes, headdir_ind):
    bins = np.linspace(0,360,11) ##bins of width 36 degrees
    binned_data = np.digitize(headdir_ind,bins) ## binned data in bins as defined above
    unique_occ, counts_occ = np.unique(binned_data, return_counts=True) ### count frequencies of occurences of each bin
    
    spike_bins = binned_data[spikes>0] ## filter by when spikes occurred
    unique_fire, counts_fire = np.unique(spike_bins, return_counts=True) ## count frequencies of bins during spikes
    
    ## compute occupancy normalized firing rates (i.e. divide firing rates by time spent looking in that direction)
    occupancy_norm = np.zeros(11)
    for i in unique_fire:
        index = np.where(unique_fire == i)[0][0]  ##index in firing rate array
        firing_rate = counts_fire[index]
        occupancy = counts_occ[i-1]
        occupancy_norm[i-1] = firing_rate/occupancy
            
    # midpoints = array([ 18.,  54.,  90., 126., 162., 198., 234., 270., 306., 342.])
    midpoints = np.zeros(11)
    for i in range(11):
        midpoints[i] = 18 + 36*i
        
    x_sum = 0
    y_sum = 0
    n = len(binned_data)
    for i in range(len(midpoints)):
        a = math.radians(midpoints[i])
        f = occupancy_norm[i]
        x_sum += f*np.cos(a)
        y_sum += f*np.sin(a)
    
    x = x_sum
    y = y_sum
    r = np.sqrt(x**2 + y**2)
    c = (36*math.pi/360)/np.sin(math.radians(36/2))  ## correction factor for grouped data
    
    return r*c


        

    
    

