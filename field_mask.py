#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 09:12:38 2020

@author: marianne
"""

import numpy as np
import matplotlib.pyplot as plt
#from ocam_camera_model_tools import OcamCalibCameraModel,vec3_normalise
from rectilinear_camera_model_tools import RectilinearCameraModel
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from scipy import interpolate
import cv2
from transformation_utilities import transform_xyz_point
from geometric_utilities import line_XY_intersection, orient2d
import time
    
class Polygon():
    '''
    Polygon represented by list of points (corners)
    Points must be in counter clockwise order
    '''
    
    def __init__(self,points,label=1):
        self.label = label
        self.points = points
        
    def plot(self):
        #plt.figure()
        plt_indeces = np.append(range(self.points.shape[0]),0)
        plt.plot(self.points[plt_indeces,0],self.points[plt_indeces,1])
        plt.axis('scaled')
        
    def make_pointpairs(self):
        return zip(self.points,np.roll(self.points,-1,axis=0))
    
    def check_if_inside(self,q):
        '''
        Check if a point q[x,y] or array of points q[[x0,x1,...],[y0,y1,...]] is inside the polygon. Using geometric predicate per each side of polygon, turning counter-clockwise.
        '''
        #Vectorized
        is_inside = np.full(q.shape[1], True, dtype=bool)
        for p, p_next in self.make_pointpairs():
            is_inside *= orient2d(p,p_next,q) > 0
        return is_inside
    

class PolygonMask():
    def __init__(self):
        self.list_of_polygons = [] #List of Polygon objects

    def add_polygon_to_mask(self,polygon):
        self.list_of_polygons.append(polygon)

    def check_if_points_inside_mask(self,points):
        """
        Check if array of points hits any of the polygons in mask
        """
        poly_indeces = np.zeros(points.shape[1])
        poly_labels = np.zeros(points.shape[1])
        for poly_index,poly in enumerate(self.list_of_polygons):
            is_inside = poly.check_if_inside(points)
            poly_indeces += is_inside*(poly_index + 1)
            poly_labels += is_inside*(poly.label + 1)
        
        #Decrement to start index with zero
        poly_indeces += -1
        poly_labels += -1
        return poly_indeces, poly_labels
    
#%% Core functionality
'''
Assumptions:
    Robot coordinate system: Aligned with robot base, moving with the robot. X ahead, Y to the left, Z up
    World coordinate system: The local ground truth coordinate system, aligned with ground plane
    Camera coordinate system: X right, Y down, Z into the image
'''

def set_up_field_mask(labels = [1,0,1], lane_spacing = None, crop_duty_cycle = None, widths = None, extent=5):
    #Create adjacent rectangles with row/crop labels and save as polygons field mask
    #Origo is at x = 0 and y = w/2 (in the middle of the center row)
    # label = 1 is lane = 0 is crop
    #x ahead, y to the left

    position = np.array([0,0])
    #list_of_polygons = []
    field_mask = PolygonMask()
    h = extent
    #shift to get the desired origo
    if widths is None:
        total_width = lane_spacing*(np.sum(labels) * (1-crop_duty_cycle) + np.sum(np.logical_not(labels)) * crop_duty_cycle)
    else:
        total_width = np.sum(widths)
    shift = -np.array([0,total_width/2])

    for ind,label in enumerate(labels):
        if widths is None:
            w = lane_spacing*((1-crop_duty_cycle)*label + crop_duty_cycle*int(not(label)))
        else:
            w = widths[ind]
        
        points = np.array([position + shift,
                           position + shift + np.array([h,0]),
                           position + shift + np.array([h,w]),
                           position + shift + np.array([0,w])
                           ])
        
        #list_of_polygons.append(Polygon(points,label))
        field_mask.add_polygon_to_mask(Polygon(points,label))
        position = position + np.array([0,w]) #position of next rectangle
    
    return field_mask

def project_mask_from_world_to_image(cam_model,mask_obj,T_cam_to_world,sampling_step = None):
    ''' 
    Make an image mask (from a polygon_mask) in world coordinates, based on camera model and camera to world transform
    Cropped dims is height, width
    Output: Mask mage with 2 channels
        channel 0: polygon index
        channel 1: label
    '''
    if sampling_step is None:
        sampling_step = 1

    camera_origo = transform_xyz_point(T_cam_to_world,[0,0,0]) #camera origo in world coordinates (compute outside loop to save time)
    
    #Generate mask image
    image_dims = np.array([cam_model.height,cam_model.width])
    subsampled_dims = np.uint16(np.floor(image_dims/sampling_step))
    mask_image = np.zeros((subsampled_dims[0],subsampled_dims[1],2))

    #For each pixel, check where it hits
    ground_points = np.zeros((2,subsampled_dims[0],subsampled_dims[1]))
    #t_gp = time.time()
    for j in np.arange(subsampled_dims[1]):
        for i in np.arange(subsampled_dims[0]):
            pixel_yx = np.array([i,j])*sampling_step
            ground_points[:,i,j] = project_pixel_to_ground(cam_model, pixel_yx, T_cam_to_world,camera_origo)
    #print("Ground point projection: ", time.time()-t_gp)
    #t_c = time.time()
    mask_indices, mask_labels = mask_obj.check_if_points_inside_mask(ground_points.reshape(2,-1))
    #print("Check if points inside mask", time.time()-t_c)
    
    mask_image[:,:,0] = mask_indices.reshape(subsampled_dims)
    mask_image[:,:,1] = mask_labels.reshape(subsampled_dims)
    return mask_image

def project_pixel_to_ground(cam_model, pixel_yx, T_cam_to_world,camera_origo):
    #Compute projected ground point from pixel
    v = cam_model.pixel_to_vector(pixel_yx[1],pixel_yx[0])
    v_world = transform_xyz_point(T_cam_to_world,v)
    gp = line_XY_intersection(point=camera_origo,direction = np.array(v_world)-np.array(camera_origo))
    return gp


if __name__ == "__main__":
    #%% Demo code: How to make an image field mask
    
    #Dummy field mask
    polygon_field_mask = set_up_field_mask(widths = [0.5,0.3,0.5], labels = [1,0,1], extent = 5)
    
    #%%Transformations 
    #Old camera model with adjustments for testing
    calib_file = os.path.join('data/realsense_model.xml')
    #cam_model = OcamCalibCameraModel(calib_file)
    cam_model = RectilinearCameraModel(calib_file)

    #Camera setup    
    #dummy values, should get values from outside
    camera_tilt = np.pi/8 
    camera_height = 1
    T_camera_to_robot = set_up_camera_to_robot_transform(rpy = [0,-camera_tilt,0], xyz =[0,0,camera_height])    
    
    
    #Robot position 
    #dummy values, should get values from outside
    robot_rpy = [0,0,0]
    robot_xyz = [0,0,0]
    T_robot_to_world = set_up_robot_to_world_transform(rpy = robot_rpy, xyz = robot_xyz)
    
    T_cam_to_world = camera_to_world_transform(T_camera_to_robot, T_robot_to_world)
    image_mask = project_mask_from_world_to_image(cam_model, polygon_field_mask, T_cam_to_world)
        
    plt.figure(10)
    plt.imshow(image_mask[:,:,0])  
    plt.figure(11)          
    plt.imshow(image_mask[:,:,1])

        
            
