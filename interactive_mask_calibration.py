import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import os
import pandas as pd

from field_mask import set_up_field_mask
from run_field_mask import label_mask_from_robot_pose
from rectilinear_camera_model_tools import RectilinearCameraModel
from ocam_camera_model_tools import OcamCalibCameraModel
from visualization_utilities import blend_color_and_image
from transformation_utilities import set_up_camera_to_robot_transform

frame = '00036'
data_dir = r'/home/marianne/Code/agriNet/data/Frogn_field'#r'./local_data/'
rec_name = '20201003_L5_N_straight-basler'#'20201003_L5_S_straight'

image_path = os.path.join(data_dir,rec_name,rec_name + '_' + frame +'.png')
camera_im = plt.imread(image_path)

sampling_step = 8
calib_file = '/home/marianne/Data/Frogn_field/basler_2019-09-30-ocam_calib.xml'#'/home/marianne/catkin_ws/src/vision-based-navigation-agri-fields/thorvald_data_extraction/camera_data_collection/realsense_model.xml'

offset_file = None #os.path.join(data_dir,rec_name,'offsets_sync.csv')

#--- Set up autolabel

#Mask setup
crop_duty_cycle = 0.53#0.65
lane_spacing = 1.25
polygon_field_mask = set_up_field_mask(lane_spacing = lane_spacing, crop_duty_cycle = crop_duty_cycle, labels = [0,1,0,1,0], extent = 5)

#Camera model file (intrinsics)
cam_model = OcamCalibCameraModel(calib_file)#RectiLinearCameraModel(calib_file)

#Camera extrinsics
camera_xyz = np.array([0, 0, 0.89]) #zero y offset
camera_rpy = np.array([0.000, np.deg2rad(-24), 0.0]) #adjusted
T_camera_to_robot = set_up_camera_to_robot_transform(rpy = camera_rpy, xyz = camera_xyz)


#--- Make initial mask
if offset_file is not None:
    offsets = pd.read_csv(offset_file)
    angular_offset = offsets['AO'][frame]
    lateral_offset = offsets['LO'][frame]
angular_offset = 0
lateral_offset = 0
robot_rpy = [0,0,angular_offset] #compensate for wrong sign (something in the transformations)
robot_xyz = [0,lateral_offset,0]
label_mask = label_mask_from_robot_pose(cam_model, polygon_field_mask, robot_rpy, robot_xyz,T_camera_to_robot,sampling_step)
overlay_im = blend_color_and_image(camera_im,label_mask,color_codes = [[None,None,None],[0,0,255],[255,255,0]],alpha=0.85) 

#--- Initialize figure
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.4)
im_vis = plt.imshow(overlay_im)
plt.axis('off')

#--- Set up sliders and callbacks
ax_yaw = plt.axes([0.2, 0.10, 0.7, 0.03])
sl_yaw = Slider(ax_yaw, r'Robot yaw', -30, 30, valinit=np.rad2deg(robot_rpy[2]))

ax_lat = plt.axes([0.2, 0.15, 0.7, 0.03])
sl_lat = Slider(ax_lat, r'Robot lateral', -0.5, 0.5, valinit=robot_xyz[1])

ax_crop = plt.axes([0.2, 0.20, 0.7, 0.03])
sl_crop = Slider(ax_crop, r'Crop width', 0, 1, valinit = crop_duty_cycle)

ax_cam_tilt = plt.axes([0.2, 0.25, 0.7, 0.03])
sl_cam_tilt = Slider(ax_cam_tilt, r'Camera tilt', -60, 0, valinit = np.rad2deg(camera_rpy[1]))

ax_cam_height = plt.axes([0.2, 0.3, 0.7, 0.03])
sl_cam_height = Slider(ax_cam_height, r'Camera height', 0, 2, valinit = camera_xyz[2])

ax_cam_arm = plt.axes([0.2, 0.35, 0.7, 0.03])
sl_cam_arm = Slider(ax_cam_arm, r'Camera arm', 0, 2, valinit = camera_xyz[0])


def update_and_draw_mask():
    label_mask = label_mask_from_robot_pose(cam_model, polygon_field_mask, robot_rpy, robot_xyz,T_camera_to_robot,sampling_step)
    #t_b = time.time()
    overlay_im = blend_color_and_image(camera_im,label_mask,color_codes = [[None,None,None],[0,0,255],[255,255,0]],alpha=0.85) 
    im_vis.set_data(overlay_im)
    fig.canvas.draw_idle()

def update(val):
    robot_rpy[2] = np.deg2rad(sl_yaw.val)
    robot_xyz[1] = sl_lat.val

    update_and_draw_mask()

def update_crop(val):
    global polygon_field_mask
    polygon_field_mask = set_up_field_mask(lane_spacing = lane_spacing, crop_duty_cycle = val, labels = [0,1,0,1,0], extent = 5)
    
    update_and_draw_mask()

def update_camera_extrinsics(val):
    tilt = np.deg2rad(sl_cam_tilt.val)
    height = sl_cam_height.val
    arm = sl_cam_arm.val
    camera_xyz[2] = height
    camera_xyz[0] = arm
    camera_rpy[1] = tilt #adjusted
    global T_camera_to_robot
    T_camera_to_robot = set_up_camera_to_robot_transform(rpy = camera_rpy, xyz = camera_xyz)

    update_and_draw_mask()

sl_yaw.on_changed(update)
sl_lat.on_changed(update)
sl_crop.on_changed(update_crop)
sl_cam_tilt.on_changed(update_camera_extrinsics)
sl_cam_height.on_changed(update_camera_extrinsics)
sl_cam_arm.on_changed(update_camera_extrinsics)

plt.show()

#--- Print or save slider values?
print('Yaw, lat,crop, tilt, height, arm: ', sl_yaw.val, sl_lat.val, sl_crop.val, sl_cam_tilt.val, sl_cam_height.val, sl_cam_arm.val)
#corrections and values

new_vals = {}
new_vals['frame'] = frame
new_vals['angular_offset_new'] = np.deg2rad(sl_yaw.val)
new_vals['lateral_offset_new'] = sl_lat.val

new_vals['yaw_corr'] = np.deg2rad(sl_yaw.val) - angular_offset
new_vals['lat_corr'] = sl_lat.val - lateral_offset
new_vals['crop_width_new'] = sl_crop.val
new_vals['cam_tilt_new']= sl_cam_tilt.val
new_vals['cam_height_new'] = sl_cam_height.val
new_vals['cam_arm_new'] = sl_cam_arm.val

import json
with open(os.path.join(data_dir,rec_name,'autolabel_calibrated_values_frame'+str(frame)),'w') as f:
    json.dump(new_vals,f)