import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

import os

from field_mask import set_up_field_mask
from run_field_mask import label_mask_from_robot_pose
from rectilinear_camera_model_tools import RectilinearCameraModel
from visualization_utilities import blend_color_and_image
from transformation_utilities import set_up_camera_to_robot_transform


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

if __name__ == "__main__":
    im_file = os.path.join('data/straight.png')
 
    camera_im = plt.imread(im_file)

    sampling_step = 8
    calib_file = '/home/marianne/catkin_ws/src/vision-based-navigation-agri-fields/thorvald_data_extraction/camera_data_collection/realsense_model.xml'

    #--- Set up autolabel

    #Mask setup
    crop_duty_cycle = 0.49#0.55#0.65
    lane_spacing = 1.25
    polygon_field_mask = set_up_field_mask(lane_spacing = lane_spacing, crop_duty_cycle = crop_duty_cycle, labels = [0,1,0,1,0], extent = 5)

    #Camera model file (intrinsics)
    cam_model = RectilinearCameraModel(calib_file)

    #Camera extrinsics
    camera_xyz = np.array([0, 0.033, 0.9])#np.array([0, 0.033, 1.1]) #zero y offset
    camera_rpy = np.array([0.000, np.deg2rad(-22.15), 0.0])  #np.array([0.000, np.deg2rad(-22.15), 0.0]) #adjusted
    T_camera_to_robot = set_up_camera_to_robot_transform(rpy = camera_rpy, xyz = camera_xyz)

    #--- Make initial mask
    robot_rpy = [0,0,0]
    robot_xyz = [0,0,0]
    label_mask = label_mask_from_robot_pose(cam_model, polygon_field_mask, robot_rpy, robot_xyz,T_camera_to_robot,sampling_step)
    overlay_im = blend_color_and_image(camera_im,label_mask,color_codes = [[None,None,None],[0,0,255],[255,255,0]],alpha=0.85) 

    #--- Initialize figure
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.4)
    im_vis = plt.imshow(overlay_im)
    plt.axis('off')

    #--- Set up sliders and callbacks
    ax_yaw = plt.axes([0.2, 0.10, 0.7, 0.03])
    sl_yaw = Slider(ax_yaw, r'Robot yaw', -30, 30, valinit=0)

    ax_lat = plt.axes([0.2, 0.15, 0.7, 0.03])
    sl_lat = Slider(ax_lat, r'Robot lateral', -0.5, 0.5, valinit=0)

    ax_crop = plt.axes([0.2, 0.20, 0.7, 0.03])
    sl_crop = Slider(ax_crop, r'Crop width', 0, 1, valinit = crop_duty_cycle)

    ax_cam_tilt = plt.axes([0.2, 0.25, 0.7, 0.03])
    sl_cam_tilt = Slider(ax_cam_tilt, r'Camera tilt', -60, 0, valinit = np.rad2deg(camera_rpy[1]))

    ax_cam_height = plt.axes([0.2, 0.3, 0.7, 0.03])
    sl_cam_height = Slider(ax_cam_height, r'Camera height', 0, 2, valinit = camera_xyz[2])

    ax_cam_arm = plt.axes([0.2, 0.35, 0.7, 0.03])
    sl_cam_arm = Slider(ax_cam_arm, r'Camera arm', 0, 2, valinit = camera_xyz[0])

    sl_yaw.on_changed(update)
    sl_lat.on_changed(update)
    sl_crop.on_changed(update_crop)
    sl_cam_tilt.on_changed(update_camera_extrinsics)
    sl_cam_height.on_changed(update_camera_extrinsics)
    sl_cam_arm.on_changed(update_camera_extrinsics)

    plt.show()

    print('Yaw, lat,crop, tilt, height, arm: ', sl_yaw.val, sl_lat.val, sl_crop.val, sl_cam_tilt.val, sl_cam_height.val, sl_cam_arm.val)

    