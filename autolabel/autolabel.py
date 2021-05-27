import numpy as np

from PIL import Image
from autolabel.field_mask import project_mask_from_world_to_image
from autolabel.transformation_utilities import set_up_robot_to_world_transform, camera_to_world_transform

def get_mask(cam_model, polygon_field_mask, robot_rpy = [0,0,0], robot_xyz = [0,0,0],T_camera_to_robot = np.eye(4), sampling_step = 1):
    '''
    Compute segmentation mask for crop row based on robot pose and camera projection
    
    Parameters
    ---
    cam_model : CameraModel object 
    polygon_field_mask : Field mask object
    robot_rpy : robot roll, pitch, yaw relative to row centerline, in radians
    robot_xyz : robot x,y,z relative to row center, in meters

    Returns
    ---
    label_mask : image with labelled pixels

    Note on coordinate systems
    --- 
    Robot coordinate system: Aligned with robot base, moving with the robot. X ahead, Y to the left, Z up
    World coordinate system: The local ground truth coordinate system, aligned with ground plane
    Camera coordinate system: X right, Y down, Z into the image

    '''
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