#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 09:12:38 2020

@author: marianne
----
Demo: Generate a field label mask and plot together with image
"""

import numpy as np
import matplotlib.pyplot as plt
import os

from autolabel.camera_models import RectilinearCameraModel
from autolabel.field_mask import set_up_field_mask, get_mask
from autolabel.transformation_utilities import set_up_camera_to_robot_transform
from autolabel.visualization_utilities import blend_color_and_image

#--Input data
im_file = os.path.join('data/slalom.png')

# Robot pose relative to row centreline at time of image capture (in meters and radians)
lateral_offset = 0.2309212808500685
angular_offset = -0.08151692116980125 

# Camera calibration 
calib_file = os.path.join('data/realsense_model.xml')

# Camera pose (found by calibration)
camera_tilt = np.deg2rad(22.15) #in radians, positive downwards
camera_offset = -0.033 #sideways shift in meters
camera_height = 0.96 #in meters above ground

# Field properties
crop_duty_cycle = 0.48 #crop width relative to total width
lane_spacing = 1.25 #"Wheel spacing", total width from lane center to lane center

#--Run

# Field mask geometry
polygon_field_mask = set_up_field_mask(lane_spacing = lane_spacing,
                                       crop_duty_cycle = crop_duty_cycle,
                                       labels = [0,1,0,1,0],
                                       extent = 5
                                       ) 
   
# Camera model
cam_model = RectilinearCameraModel(calib_file)

# Camera extrinsics
T_camera_to_robot = set_up_camera_to_robot_transform(rpy = np.array([0.0, -camera_tilt, 0.0]), 
                                                    xyz = np.array([0,camera_offset,camera_height])
                                                    )    

# Project mask
label_mask = get_mask(cam_model, 
                                polygon_field_mask, 
                                robot_rpy = [0,0,angular_offset], 
                                robot_xyz = [0,lateral_offset,0],
                                T_camera_to_robot = T_camera_to_robot,
                                sampling_step = 2) #image subsampling for speedup

# Visualise on image
camera_im = plt.imread(im_file)
overlay_im = blend_color_and_image(camera_im,
                                  label_mask,
                                  color_codes = [[None,None,None],[255,255,0],[255,0,255]], #color codes for background, lane and crop
                                  alpha=0.75) 

# Plot
plt.figure()
plt.imshow(label_mask)
plt.axis('off')
plt.figure()
plt.imshow(overlay_im)
plt.axis('off')

plt.show()