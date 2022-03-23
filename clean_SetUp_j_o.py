#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 13:14:00 2021

@author: sabinechavin
"""

import time
import numpy as np
import os
from scipy import stats, io, signal,interpolate


def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]

#### Returns x,y,x_ind,y_ind, headdir_ind, x_of_bins,y_of_bins, spatial_scale, boundaries, time_ind, traj_filename"""
## x,y = x and y coordinates at each time step
## headdir_ind = head direction angle at each time step
## x_of_bins,y_of_bins = x and y positions of each bin (cm)
## spatial scale = number to scale by (not sure where this comes from, is it total len?)
## time_ind = number of indices, and thus number of times model will run

def get_trajectory(filename, dt):
    'Pulled from grid code, do not need velocity calculations for now so removed these'
    'Add velocity back in if required later'
    '**More pressing** add update that take input argument file you want to look for trajectories and what index in that file you want to take'
    
    print('getting trajectory data from Hass lab')
    
    traj_filename = filename
    unpack=io.loadmat(traj_filename)
    times = 17*60; #hardcoded because currently using only 17 minutes of each session
    traj_fs = unpack['traj_fs']
    
    sesh_ts_og = np.arange(0,times,1/traj_fs); #time stamps (in seconds) for positions with original sampling rate
    sesh_ts_interp = np. arange(0,(times),(dt/1000)); #new time stamps (in seconds) for dt of model sampling rate
    
    spatial_scale = unpack['traj_spatial_scale']
    x = unpack['traj_x']*spatial_scale;#position in cm, below is in indecies, resample up to dt samples
    y = unpack['traj_y']*spatial_scale; #position in cm, below is in indecies, resample up to dt samples
    headdir = unpack['traj_hd']

    x = x[0:np.size(sesh_ts_og)+1]
    y = y[0:np.size(sesh_ts_og)+1]
#    headdir = headdir[0:np.size(sesh_ts_og)+1]
    z = headdir[0:np.size(sesh_ts_og)+1]
    #just to make sure we are all equal
    sesh_ts_og= sesh_ts_og[0:np.size(x)]
    
    #dumb fix to adjust for interpolation issues with some values of sesh_ts_interp being outside the range of sesh_ts_og
    sesh_ts_interp[sesh_ts_interp>np.max(sesh_ts_og)] = np.max(sesh_ts_og)
    t = [num[0] if not np.isnan(num) else 180*np.random.rand(1)[0] for num in z]
    
    if any(np.isnan(x)) or any(np.isnan(y)):
        
        nans1, t1= nan_helper(x)
        nans2, t2= nan_helper(y)
    #    nans3, t3 = nan_helper(z)
        
        x[nans1]= np.interp(t1(nans1), t1(~nans1), x[~nans1])
        y[nans2]= np.interp(t2(nans2), t2(~nans2), y[~nans2])
    #    z[nans3] = np.interp(t3(nans3), t3(~nans3), z[~nans3])
        #shouldn't be any nans in headdir, leaving it out for now to check this fact
    
#    note hardcoded ms, also note change from before
    x_ind = np.round(signal.resample(x/spatial_scale,int(times/(dt/1000)))) #resample up to dt samples
    x_ind = x_ind.astype('int');
    y_ind = np.round(signal.resample(y/spatial_scale,int(times/(dt/1000)))) # resample up to dt samples
    y_ind = y_ind.astype('int');

#    headdir_ind = signal.resample(headdir,int(times/(dt/1000))) #resample up to dt samples, don't round
    headdir_ind = signal.resample(t,int(times/(dt/1000))) #resample up to dt samples, don't round
  #  print(np.isnan(np.sum(headdir_ind)))
    
    '****this is another thing we will have to work to fix going forward.  Should probably just record walls for environment ahead of time'
    'Also, will note this in other sections, but need to decide on matrix vs list implementation'
    '-matrix is easier/works well for boxes, but in the future with weirder environments we may just want to do a precalculation of distance'
    #Use max recorded position to do box walls for now; note: given implementation sketched in notes, will need to switch this to overall max to get square to then circumscribe the environment
    dim_box = np.array([np.max(x), np.max(y)])#np.max([np.max(x), np.max(y)]) #<<<<'this is og setting'
    
    boundaries = ([0,0],[dim_box[0], dim_box[1]])
    'note need to address this issue as well, but still not a clear answer on how to center what is zero, zero across environment given trajectory'
    #double check if zero makes sense in all cases...might be aligning the border cells on the south wall too low and thus their firing fields are smaller than the should be
    #update to aboveL zero is definitely an imperfect call as doesn't fully capture the environment, not positive how to fix though as it is the minimum...
    x_of_bins = np.arange(np.min(boundaries,0)[0],np.max(boundaries,0)[0]+spatial_scale,spatial_scale)
    y_of_bins = np.arange(np.min(boundaries,0)[1],np.max(boundaries,0)[1]+spatial_scale,spatial_scale)
    
    if np.any(x_ind>=np.size(x_of_bins)) or np.any(y_ind >= np.size(y_of_bins)):
        
        x_ind[x_ind>=np.size(x_of_bins)] = np.size(x_of_bins)-1;
        y_ind[y_ind>=np.size(y_of_bins)] = np.size(y_of_bins)-1; #update: this was incorrectly referencing x_ind in the index call
     #note can shorten how much of even this slice of the trajectory you run by changing time_ind   
    time_ind = int(times/(dt/1000)); #simply the number of indecies and therefore iterations for the model to run
    
    return x,y,x_ind,y_ind, headdir_ind, x_of_bins,y_of_bins, spatial_scale, boundaries, time_ind, traj_filename


## returns a vector of length 8 where each position in vector is either:
    ### -1 to indicate no wall within threshold
    ### distance from wall in cm
    ### position 0 = 0 deg, position 1 = 45 deg and so on... position 7 = 315 deg 
    ## angles measured clockwise from 0, see Gofman paper image for reference

def get_egostates(x_ind,y_ind,headdir_ind, x_of_bins,y_of_bins,boundaries,spatial_scale,time_ind,traj_filename,max_dis_threshold):
    'gets all relevant states (allo and ego distance from wall, allo angle, etc)  given trajectory'
    'Note as currently implemented, distances are all in bins (i.e. number of spatial bins/pixels) rather than cm'
    'That may change, but probably better to deal in ints so dont have to be as accurate as floats'
    'Changed mind need to load in HD from file and do interp so stats are correct.'

    ego_walldis = np.ones([48,time_ind])*-1 # 24 angles, increments of 15
    
    #i.e. 0 to 315

    #Doing calculation based on distance from current position, see if any are less than threshold (just make threshold equal to max),
    #get allo angles for all that are less than threshold, then rotate to ego angles and fille ego_walldis 
    
    #For now do this to calculate boundary points.  Will have to do something else later when have more complex environments.
    #Just doing this way for speed can clean up later if needed
    #Keeping this as inds/bins throughout for now (I.e. units are in cms yet)
    

    #North Wall
    North = np.array([np.arange(0,x_of_bins.size), np.ones([x_of_bins.shape[0]])*y_of_bins.size]) #length of x at y max
    
    #East Wall
    East = np.array([np.ones([y_of_bins.shape[0]])*x_of_bins.size,np.arange(0,y_of_bins.size)]) #length of y at x max
    
    #South Wall
    South = np.array([np.arange(0,x_of_bins.size), np.zeros([x_of_bins.shape[0]])]) #length of x at y min
    
    #West Wall
    West = np.array([np.zeros([y_of_bins.shape[0]]),np.arange(0,y_of_bins.size)]) #length of y at x min
    
    boundary_points = np.concatenate((North,East,South,West), axis = 1) #Now just concatenate them all together
    
    boundary_points = boundary_points.astype('int') #Since everything is bin don't need to have float accuracy, can stick with ints

    
    #Working to not have to use for loop, but may be unavoidable since using so much ram right now
    #UPDATE: tried to not use for loop but think this will destory ram otherwise, not going to do this online as would like to be able to use this
    #function to get this info even if not running a simuliation.  May take more time, but can also save output and only run if this hasn't been already calculated
    
    
    # #Get displacement from current point to all boundary points at each time point
    # dis_hold = np.zeros([2,time_ind,boundary_points.shape[1]],dtype = 'int32') #preallocate
    
    # #May be issue for less ram heavy computers as this peaks us out right now
    # dis_hold[0,:,:] = boundary_points[0,:]-x_ind[:]
    # dis_hold[1,:,:] = boundary_points[1,:]-y_ind[:]
    
    #^Old code from when trying to avoid for loop
    
    threshold = np.round(max_dis_threshold/spatial_scale) #assumes max_dis_threshold is in cm and sets threshold to corresponding number of bins/inds
    print("thresh",threshold)
    for index in range(0,time_ind):
        
        if np.mod(index,1000)==0:
                        print(index)
                        
        centered_temp = np.array([boundary_points[0,:]-x_ind[index],boundary_points[1,:]-y_ind[index]])
        #Get distance from current point to all boundary points
        dis_temp = np.sqrt(np.square(centered_temp[0,:]) + np.square(centered_temp[1,:]))
        
        #See if any distance is below threshold set above
        check = dis_temp <=threshold[0][0]
        dis_holding = dis_temp[check] #centered_temp[:,check]#change after debug just for comparing with allo_angles after arctan2 to make sure everything makes sense
        #Get angle for all that are within threshold
        allo_angles = np.arctan2(centered_temp[1,check],centered_temp[0,check])
        allo_angles = np.rad2deg(allo_angles)
        allo_angles = allo_angles - 90.0 #Do rotation to make North 0 degrees.
        #Set to be on same scale as headdir variable, -180 to +180, by adding 360 to all angle LESS THAN <-180 (i.e. -180 to -360)
        
        allo_angles[allo_angles<=-180] = allo_angles[allo_angles<=-180]+360
        ego_angles = allo_angles - headdir_ind[index] #This should work since now both allo_angles and headdir_ind are on the same scale
        ego_angles = ego_angles *-1.0 #Do sign flip so that right is 90 degrees and left is negative 90 degrees
        ego_angles[ego_angles<0] = ego_angles[ego_angles<0]+360 #after flip make all angles positive to align with Goffman paper
        
        #Get indecies for whatever points using np.where(np.any(condition,axis = 1))
        #looks like it works from first point with this data, which is near East wall
        #also works with index 6192 which is next to South wall, added rotation to make sure west is 90 and approaching south from west is +90 to +180
        #West test at index 24033 works, recall east and west want y zeroed and north and south want x zeroed
        #North test at index 35920 works
        #Now have 0 is north, -90 is east, +/-180 is South, and 90 is west
        
        #Repeat checks to make sure allo to ego works especially with sign flip
        #East check works,closest wall is -90 east, animal is running due east, North should be negative angles due to flip since on left, south should be positive angles due to flip since on right, 
        #South check , closest wall is 180 due south, animal is running South East (-148), ego angle should be small positive angle since south will be on animals right, angles should then flip with east being large positive angles in ego space as east is on animals left and west is on animals right
        
        #Note: for now looks like just drawing 90 degree rays but that is fine as this code should hopefully still work in more complex environments.
        
        #Since now all positive can just do angle checks based on sorted listed of angles, potentially better way to code this but easiest thing for now
        #For now will have to manually come back and change these, but for now doing 45 degree slices centered on angle
        #angles go 0, 45, 90, 135, 180, 225, 270, 315
        #Puts in the minimum distance at angle or leaves -1 if no angle within threshold for that time step.
        #Not the best coding, but first thing that came to mind
        
        #NOTE*********************
        #Outputting final distances as cm for easy interaction with setting receptive field
        
        #only update from -1 if angles are within threshold

        if np.any(np.logical_or(ego_angles < 15, ego_angles >= 345)):
           ego_walldis[0,index]= np.min(dis_holding[np.logical_or(ego_angles < 15, ego_angles >= 345)])*spatial_scale
        
        for i in range(1,48):
            center = i * 7.5
            min_val = center - 15
            max_val = center + 15
            
            
            if np.any(np.logical_and(ego_angles >= min_val, ego_angles < max_val)):
                ego_walldis[i,index] = np.min(dis_holding[np.logical_and(ego_angles >= min_val, ego_angles < max_val)])*spatial_scale
     

        #IMPORTANT UPDATE TO DO
        #after get basic case working definitely save the output from this process in the trajectory folder and check whether this file exists in main to save time
        
    #saving the output of this goes here
    cwd = os.getcwd() #get where you currently are (i.e. the current directory) #change this later
        
    var_out = dict()
    var_out['ego_walldis'] = locals()['ego_walldis']
    
    io.savemat('second_test_n=48_scale.mat' ,var_out)
        
        #Also note, for now just returning ego_walldis, but could add indexing to any of the above variables if we need those outputted
    return ego_walldis


# returns weight matrices populated with numbers between 0 and rand_weights_max
# returns  w_EBCx_EBC_HD, w_EBCxHD_HD, w_EBCxHD_EBC, w_ABC_EBCxHD
def weights(out_cells, in_cells, rand_max):
    w = np.reshape(stats.uniform.rvs(0,rand_max,out_cells*in_cells),(in_cells,out_cells))
    return w
    
def setup_HD_fields(n_HD):
# Set up receptive fields of HD cells
    params = 1
    HD_fields = np.zeros((n_HD,params))
    HD_options = np.array([0,45,90,135,180,-90,-45, -135]) #Update making HD tuning 90 degree tunings
    HD_options = np.tile(HD_options,n_HD+1)
    HD_fields[:,0] = HD_options[0:n_HD]
    if n_HD == 4:
        HD_fields[:,0] = np.array([-90,0,90,180])
    if n_HD == 2:
        HD_fields[:,0] = np.array([180,180])
    if n_HD == 16:
        #HD_fields[:,0] = np.array([0,,0,0,90,90,90,90,180,180,180,180,-90,-90,-90,-90])
        HD_fields[:,0] = np.linspace(0,360, n_HD)
    if n_HD == 32:
        tot = np.append(np.linspace(0,360, 16),(np.linspace(0,360, 16)))
        HD_fields[:,0] = tot
        
    if n_HD == 96:
        tot = np.append(np.linspace(0,360, 48),(np.linspace(0,360, 48)))
        HD_fields[:,0] = tot
        
        
    return HD_fields

def setup_EBC_fields(n_EBC, max_dis_threshold):
    
    params = 2
    EBC_fields = np.zeros((n_EBC,params))
    EBC_fields[:,0] = np.ones((n_EBC))*max_dis_threshold
    options = list(np.arange(0,48))
    EBC_options = np.tile(options,n_EBC+1)  ##options for potential EBC fields, so far only cardinal directions
    EBC_fields[:,1] = EBC_options[0:n_EBC]
    
    return EBC_fields

def setup_EBCxHD_fields(n_EBCxHD, max_dis_threshold):
    #for EBCxHD need to pick distribution of HD angles and Ego-boundary angles as well as distances
    #for now no parameters for tuning sharpness of the fields, will leave that for the update code
    params = 3 
    EBCxHD_fields = np.zeros((n_EBCxHD,params))
    
    #distance from wall threshold will all be the same for now
    EBCxHD_fields[:,0] = np.ones((n_EBCxHD))*max_dis_threshold
    
    #ego boundary angles have to be split between 8 central angles, for now distribution will be even
    #i.e. this simply tells you which of the 8 state values to grab.  can think about making this multiple values later
    #again that is probably when we should consider making each cell in the network a class that can have its own attributes

    ego_options = np.tile(np.arange(0,8),round(n_EBCxHD/8)) #will need to add +1 in case rounds down when use numbers not divisible by 8
    
    np.random.shuffle(ego_options) #note shuffle works in place so don't need to reassign
    
    #just add as many as can in case n_EBCxHD is not divisible by 8, add pay attention to above note about adding 1

#    EBCxHD_fields[:,1] = ego_options[0:n_EBCxHD]
        
    #HD angles will be same as ego boundary angles for now, i.e. evenly distributed across 8 central angles
    #later on can just pull from desired distribution across angles
    #Putting in as negative angle so they match the headdir variable we get from data, put in toggle for this based on traj file at some point
    
    HD_options = np.array([0,45,90,135,180,-45,-90,-135]) #Update making HD tuning 90 degree tunings
    HD_options = np.tile(HD_options,round(n_EBCxHD/HD_options.size))
    np.random.shuffle(HD_options)
        
    #EBCxHD_fields[:,2] = HD_options[0:n_EBCxHD]
    return EBCxHD_fields

def setup_place_fields(n_place, dim, radius):
    
    ## 3 params are x,y and radius
    params = 4
    place_fields = np.zeros((n_place,params))
    xs = np.random.uniform(0, dim, size = n_place)
    ys = np.random.uniform(0, dim, size = n_place)
    radii = np.ones((n_place))*radius
    box_indices = []
    
    for i in range(n_place):
        x = xs[i]
        y = ys[i]
        xbin = round(x/75)
        ybin = round(y/75)
        box = xbin * 4 + 4 * ybin
        box_indices.append(box)
        
    place_fields = np.zeros((n_place,params))
    place_fields[:,0] = xs
    place_fields[:,1] = ys
    place_fields[:,2] = radii
    place_fields[:,3] = box_indices
    
    return place_fields
    

## return a 2d grid where each position corresponds to place cell
## at each position in grid there is either:
    ### -1 if there are no place cells in that position 
    ### add index of place cell to list stored at position in grid
def setup_place_grid(n_place, dim, radius):
        
    rows, cols = (dim, dim)
    arr = [[-1 for i in range(cols)] for j in range(rows)]
    xs = np.random.randint(0,dim,size=n_place)
    ys = np.random.randint(0,dim,size=n_place)
    for i in range(n_place):
        x = xs[i] ## coordinates of center of place cell
        y = ys[i]

        ## for now just square place cells
        for h in range(-radius,radius):
            #print(h)
            for w in range(-radius,radius):
                xpos = x+h
                ypos = y+w
                if xpos<dim and ypos<dim and xpos>=0 and ypos>=0:
                    v =  arr[xpos][ypos]
                    if v == -1:
                       arr[xpos][ypos] = [i]
                    else:
                        v.append(i)
                        arr[xpos][ypos] = v
                """
                xpos = x-h
                ypos = y-w
                
                if xpos>=0 and ypos>=0:
                    v =  arr[xpos][ypos]
                    if v == -1:
                        arr[xpos][ypos] = i
                 #   else:
                      #  if not(i in v):
                       #     v.append(i)
                        #    arr[xpos][ypos] = v
                        """
    print("place array initialized")                        
    return arr




