import numpy as np

#%% General transformation stuff
'''
Roll, pitch, yaw is rotation around x,y,z axis. 
To combine rotation matrices, rotate in x,y,z order around x axis first: R=RzRyRx

'''

def x_rotation_matrix(theta):
    Rx = np.array([
            np.array([1, 0, 0]),
            np.array([0, np.cos(theta), np.sin(theta)]),
            np.array([0, -np.sin(theta), np.cos(theta)])
            ])
    return Rx

def y_rotation_matrix(theta):
    Ry = np.array([
            [np.cos(theta), 0, -np.sin(theta)],
            [0, 1, 0],
            [np.sin(theta), 0, np.cos(theta)]
            ])
    return Ry
    
def z_rotation_matrix(theta):
    Rz = np.array([
            [np.cos(theta), np.sin(theta), 0],
            [-np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
            ])
    return Rz

def create_transformation_matrix(r,t):
    '''
    Make a homogeneous 4x4 translation matrix from rotation angles (in radians) rx,ry,rz and translations (in meteres) tx,ty,tz
    Transformation order:
        1. x axis rotation
        2. y axis rotation
        3. z axis rotation
        4. translation
    '''
    rx,ry,rz = r 
    tx,ty,tz = t
    Rx = x_rotation_matrix(rx)
    Ry = y_rotation_matrix(ry)
    Rz = z_rotation_matrix(rz)
    R = Rz.dot(Ry).dot(Rx) #combine rotation matrices: x rotation first
    t = np.array([tx,ty,tz])
    T = np.eye(4)
    T[0:3,0:3] = R
    T[0:3,3] = t
    return T

def transform_xyz_point(T,point):
    P = np.eye(4)
    P[0:3,3] = point
    P_transformed = T.dot(P)
    return P_transformed[0:3,3]

def camera_to_world_transform(T_camera_to_robot = np.eye(4), T_robot_to_world = np.eye(4)):
    return T_robot_to_world.dot(T_camera_to_robot)

def set_up_robot_to_world_transform(rpy, xyz):
    return create_transformation_matrix(r=rpy, t=xyz)
        
def set_up_camera_to_robot_transform(rpy = [0,0,0], xyz = [0,0,0]):
    ''' 
    Robot coordinates: x ahead, y left, z up
    Camera coordinaes: x right, y down, z ahead
    
    inputs: camera pose in robot (base) coordinates
    '''
    # Rotation between coordinate systems. Creating a camera coordinate system 
    # aligned with the robot coordinate system (x_robot = z_cam, y_robot = -x_cam, z_robot = -y_cam)

    rx = np.pi/2
    ry = 0
    rz = np.pi/2
    T_cam_to_camaligned = create_transformation_matrix(r = [rx,ry,rz],t = [0,0,0])
    #Camera tilt and position compared to robot coordinate system
    T_camaligned_to_rob = create_transformation_matrix(r = [rpy[0], rpy[1], -rpy[2]],t = xyz) #compensate for sign error
    #Combine 
    T_cam_to_rob = T_camaligned_to_rob.dot(T_cam_to_camaligned)
    return T_cam_to_rob