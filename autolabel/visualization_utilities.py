"""
Created 2020

@author: Marianne Bakken

"""
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from PIL import Image

def blend_color_and_image(image,mask,color_codes,alpha=0.5, mask_values = [0,1,2]):
    '''
    Blend colored mask and input image
    Input:
      3-channel image (numpy array)
      1-channel (integer) mask with N different values
      Nx3 list of RGB color codes (for each value mask).Example for mask with values 0,1,2:
    '''
    if mask_values is None:
        mask_values = np.unique(mask)
    color_codes = np.array(color_codes,dtype=float)
    assert(color_codes.shape[0] == len(mask_values))
    #convert input to uint8 image
    if image.dtype is np.dtype('float32') or np.dtype('float64') and np.max(image) <= 1:
        image = np.uint8(image*255)
    mask = np.tile(mask[:,:,np.newaxis],[1,1,3])
    #convert nan values to zero
    mask = np.nan_to_num(mask)
    
    #Add one layer per value (larger than zero) in mask
    blended_im = image
    for ind in range(0,len(mask_values)):
        if not np.isnan(color_codes[ind,0]):
            val = mask_values[ind]
            tmp_mask = (mask == val)
            blended_im = np.uint8((tmp_mask * (1-alpha) * color_codes[ind,:]) + (tmp_mask * alpha * blended_im) + (np.logical_not(tmp_mask) * blended_im)) 
    return blended_im

def vis_overlay(im, pr, fig = None, class_values = [0,1,2], color_codes = [[None,None,None],[255,255,0],[0,0,255]], alpha = 0.8):   
    pred_size = pr.shape 
    im_resized = cv2.resize(im, (pred_size[1], pred_size[0]), interpolation=cv2.INTER_NEAREST)
    overlay_im = blend_color_and_image(im_resized,pr,color_codes = color_codes, alpha = alpha , mask_values = class_values) 
    return overlay_im

