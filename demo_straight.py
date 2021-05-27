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
from visualization_utilities import blend_color_and_image

# Demo: Generate a field label mask and visualise on image

if __name__ == "__main__":
    #--Input data
    im_file = os.path.join('data/straight.png')

    #Robot pose relative to row centreline at time of image capture (in meters and radians)
    lateral_offset = 0
    angular_offset = 0

    #Camera calibration 
    calib_file = os.path.join('data/realsense_model.xml')

    #Camera pose (found by calibration)
    camera_tilt = np.deg2rad(22.15) #in radians, positive downwards
    camera_offset = -0.033 #sideways shift in meters
    camera_height = 0.96 #in meters above ground

    #--Setup
    sampling_step = 1
    crop_duty_cycle = 0.48#0.65
    lane_spacing = 1.25

    #--Run

    #Field mask geometry
    polygon_field_mask = set_up_field_mask(lane_spacing = lane_spacing, crop_duty_cycle = crop_duty_cycle, labels = [0,1,0,1,0], extent = 5) 
    
    #Camera model
    calib_file = os.path.join('data/realsense_model.xml')
    cam_model = RectilinearCameraModel(calib_file)

    #Camera extrinsics
    T_camera_to_robot = set_up_camera_to_robot_transform(rpy = np.array([0.0, -camera_tilt, 0.0]), xyz = np.array([0,camera_offset,camera_height]))    

    #Project mask
    label_mask = label_mask_from_robot_pose(cam_model, polygon_field_mask, robot_rpy = [0,0,angular_offset], robot_xyz = [0,lateral_offset,0],T_camera_to_robot = T_camera_to_robot ,sampling_step=4)

    #visualise
    camera_im = plt.imread(im_file)
    overlay_im = blend_color_and_image(camera_im,label_mask,color_codes = [[None,None,None],[255,255,0],[255,0,255]],alpha=0.75) 

    #Plot
    plt.figure(10)
    plt.imshow(label_mask)
    plt.figure(11)
    plt.imshow(overlay_im)

    plt.show()

        
            
