#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 09:12:38 2020

@author: marianne
"""

import numpy as np
import matplotlib.pyplot as plt
from rectilinear_camera_model_tools import RectilinearCameraModel
#from utilities import read_robot_offset_from_file, read_row_spec_from_file
import os
from field_mask import set_up_field_mask
from transformation_utilities import set_up_camera_to_robot_transform

from make_mask_from_robot_pose import label_mask_from_robot_pose #fixme change naming

if __name__ == "__main__":
    # Demo code: How to generate a field label mask

    #Field mask geometry
    polygon_field_mask = set_up_field_mask(widths = [0.5,0.3,0.5], labels = [1,0,1], extent = 5)
    
    #Camera model
    calib_file = os.path.join('data/realsense_model.xml')
    cam_model = RectilinearCameraModel(calib_file)

    #Camera extrinsics
    camera_tilt = np.pi/8 
    camera_height = 1
    T_camera_to_robot = set_up_camera_to_robot_transform(rpy = [0,-camera_tilt,0], xyz =[0,0,camera_height])    
    
    #6-DOF robot pose 
    robot_rpy = [0,0,-np.deg2rad(-10)] #roll-pitch-yaw in radians
    robot_xyz = [0,0,0] #in meters

    label_mask = label_mask_from_robot_pose(cam_model, polygon_field_mask, robot_rpy, robot_xyz,T_camera_to_robot,sampling_step=4)
        
    plt.figure(10)
    plt.imshow(label_mask)

    plt.show()

        
            
