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
from field_mask import project_mask_from_world_to_image, set_up_field_mask
from PIL import Image
import cv2
import csv
import glob
import re
#from tqdm import tqdm
import argparse
import sys
sys.path.append('..')
from visualization_utilities import blend_color_and_image
from transformation_utilities import set_up_robot_to_world_transform, set_up_camera_to_robot_transform, camera_to_world_transform
import time
import pandas as pd 

def label_mask_from_robot_pose(cam_model, polygon_field_mask, robot_rpy = [0,0,0], robot_xyz = [0,0,0],T_camera_to_robot = np.eye(4), sampling_step = 1):
    #Set up transforms
    T_robot_to_world = set_up_robot_to_world_transform(rpy = [robot_rpy[0],robot_rpy[1],-robot_rpy[2]], xyz = robot_xyz) #compensate for sign error
    T_cam_to_world = camera_to_world_transform(T_camera_to_robot, T_robot_to_world)
    
    #Project mask
    mask_with_index_and_label = project_mask_from_world_to_image(cam_model, polygon_field_mask, T_cam_to_world,sampling_step = sampling_step)

    #prepare label mask for saving and visualization
    label_mask = mask_with_index_and_label[:,:,1] #extract second channel
    label_mask = label_mask + 1 #shift from 0 and 1 to 1 and 2
    label_mask = np.nan_to_num(label_mask).astype('uint8')

    camera_im_dims = (cam_model.height, cam_model.width)
    if label_mask.shape != camera_im_dims:
        label_im = Image.fromarray(label_mask,mode='L')
        label_im = label_im.resize((camera_im_dims[1],camera_im_dims[0]))
        label_mask = np.array(label_im)
    
    return label_mask

def run_field_mask_on_files(image_dir = './local_data/20201003_L5_S_straight',
    rec_prefix = 'frame',
    output_dir = './output/automatic_annotations',
    calib_file = '/home/marianne/catkin_ws/src/vision-based-navigation-agri-fields/thorvald_data_extraction/camera_data_collection/realsense_model.xml',#'../camera_data_collection/realsense_model.xml'#os.path.join('../camera_data_collection/realsense_model_cropped.xml'),
    robot_offset_file = './local_data/20201003_L5_S_straight/offsets_sync.csv',
    use_robot_offsets = True,
    csv_delimiter = ',',
    sampling_step = 8,
    show_visualization = False,
    save_visualization = True,
    save_annotations = False):

    if show_visualization == False:
        import matplotlib
        matplotlib.use('Agg') #suppress figures

    #---Setup
    #camera model file
    cam_model = RectiLinearCameraModel(calib_file)
    #Saving dirs
    vis_dir = os.path.join(output_dir,'visualization')
    os.makedirs(vis_dir, exist_ok = True)
    ann_dir = os.path.join(output_dir,'annotation_images')
    os.makedirs(ann_dir, exist_ok = True)
    arr_dir = os.path.join(output_dir,'annotation_arrays')  
    os.makedirs(arr_dir, exist_ok = True)

    #Specify file pattern
    im_file_pattern = re.compile(rec_prefix+'_\d\d\d\d\d.png',re.UNICODE) 
    
    #Read row spec
    #Skip automatic row spec reading from file for now
    crop_duty_cycle = 0.48#0.65
    lane_spacing = 1.25

    #Camera extrinsics
    camera_xyz = np.array([0,0,0.96])#np.array([0, 0.033, 1.1]) #zero y offset
    camera_rpy = np.array([0.0, np.deg2rad(-22.15), 0.0])#np.array([0.000, -0.4, 0.0]) #adjusted

    #Correction values
    angular_correction = 0.04798 #20201003_L5_S_straight
    lateral_correction = 0#-0.1045 #20201003_L5_S_straight
    
    #Setup
    polygon_field_mask = set_up_field_mask(lane_spacing = lane_spacing, crop_duty_cycle = crop_duty_cycle, labels = [0,1,0,1,0], extent = 5) #read from file?
    T_camera_to_robot = set_up_camera_to_robot_transform(rpy = camera_rpy, xyz = camera_xyz)

    #Read offset file
    if use_robot_offsets:
        offsets = pd.read_csv(robot_offset_file,sep=csv_delimiter,encoding = 'UTF-8',index_col='frame')

    #--- For each image in the specified folder
    files = sorted(os.listdir(image_dir))
    pat = im_file_pattern
    #initialize visualization
    fg = plt.figure()
    ax = fg.gca()
    img_dim = np.zeros((60,80,3)) #hack
    h = ax.imshow(img_dim)
    ax.axis('off')
    im_files = filter(pat.match, files)

    try: 
        print('First file: ', list(im_files)[0])
    except:
        print('Could not find any files matching pattern. First file: ', list(files)[0], ', pattern:', pat)
    else:
        for im_file in tqdm(filter(pat.match, files)): 
            print(im_file)
            im_name = os.path.splitext(os.path.basename(im_file))[0]
            frame_ind = int(im_name[-4:])   
            if use_robot_offsets:
                angular_offset = offsets['AO'][frame_ind] + angular_correction
                lateral_offset = offsets['LO'][frame_ind] + lateral_correction
            else:
                angular_offset = 0
                lateral_offset = 0
            robot_rpy = [0,0,angular_offset] 
            robot_xyz = [0,lateral_offset,0] 
            label_mask = label_mask_from_robot_pose(cam_model, polygon_field_mask, robot_rpy, robot_xyz,T_camera_to_robot,sampling_step)
            #make blended image
            camera_im = plt.imread(os.path.join(image_dir,im_file))
            overlay_im = blend_color_and_image(camera_im,label_mask,color_codes = [[None,None,None],[0,0,255],[255,255,0]],alpha=0.85) 

            if save_visualization or show_visualization:
                #Save or plot
                #Plot
                h.set_data(overlay_im)
                #ax.set_title('LO: ' + str(lateral_offset) + ',AO: ' + str(np.rad2deg(angular_offset)))
                ax.set_title('Frame: {}, Lateral offset (m): {:.2f}, Angular offset (deg): {:.2f}'.format(frame_ind,lateral_offset,np.rad2deg(angular_offset)))
                plt.draw()
                if show_visualization:
                    plt.pause(1e-4)
                
                if save_visualization:
                    plt.savefig(os.path.join(vis_dir,im_name + '_overlay.png'))

            if save_annotations:
                #Save annotiations alone as images and arrays
                #plt.imsave(os.path.join(ann_dir,im_name)+'.png',label_mask)
                cv2.imwrite(os.path.join(ann_dir,im_name)+'.png',label_mask)
                np.save(os.path.join(arr_dir,im_name),label_mask)


if __name__ == "__main__":
    data_dir = './local_data'
    rec_names = ['20201003_L5_N_slaloam'] 
    sub_dir = 'all'

    for rec_name in rec_names:
        dataset_dir = os.path.join(data_dir,rec_name,sub_dir)
        run_field_mask_on_files(
            image_dir = os.path.join(dataset_dir,'images'),
            rec_prefix = rec_name,
            output_dir = os.path.join('./for_video',rec_name,sub_dir),
            calib_file = '/home/marianne/catkin_ws/src/vision-based-navigation-agri-fields/thorvald_data_extraction/camera_data_collection/realsense_model.xml',#'../camera_data_collection/realsense_model.xml'#os.path.join('../camera_data_collection/realsense_model_cropped.xml'),
            robot_offset_file = os.path.join(data_dir,rec_name,'offsets_sync.csv'),
            csv_delimiter = ',',
            sampling_step = 2,
            show_visualization = False,
            save_visualization = True,
            save_annotations = True) #normally False for debugging

