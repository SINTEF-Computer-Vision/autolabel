import numpy as np 
import matplotlib.pyplot as plt
import math
#Utility functions for two-dimensional vectors

def convert_angle_to_plus_minus_pi(angle):
    return(np.arctan2(np.sin(angle),np.cos(angle)))

def direction_sign(v0, v1):
    #Input: unit vectors in 2D
    #Check whether vector 1 has the same direction as vector 2 (the angle between them is smaller than 90 degrees)
    return np.sign(np.dot(v0,v1))

def angle_between_lines(v0,v1):
    # Input: unit vectors describing lines in 2D
    # Compute angle between lines (directionless). Output range is from -pi/2 to pi/2
    sin_theta = np.cross(v0,v1)
    return np.arcsin(sin_theta)

def angle_between_vectors(v0,v1):
    # Input: unit vectors in 2D
    # Compute angle between vectors (with direction) Output range is from -pi to pi
    sin_theta = np.cross(v0,v1)
    cos_theta = np.dot(v0,v1)
    return math.atan2(sin_theta,cos_theta)

def signed_distance_point_to_line(point, line_point, line_vector):
    #Input: 2D points, 2D unit vector
    point_vector = point-line_point
    d = np.cross(line_vector,point_vector)
    return d

def closest_point(x0,y0,xs,ys):
    #Compute distance between points, given as lists or np arrays of x and y coordinates
    x0 = np.repeat(x0,len(xs))
    y0 = np.repeat(y0,len(ys))
    sum_squared_error = 0.5*np.sqrt((x0-xs)**2 +(y0-ys)**2)
    ind = np.argmin(sum_squared_error)
    
    return ind

def line_to_next_point(point_ind, xs, ys, step = 1):
    #Line segment directly from current to next point
    #Returns line on point, unit vector form
    point = np.array([xs[point_ind],ys[point_ind]])
    next_point = np.array([xs[point_ind+step],ys[point_ind+step]])
    vector = next_point-point
    vector = vector/np.linalg.norm(vector)

    return point,vector

def line_fit_from_points(point_ind,xs,ys,forward_window=20, backward_window=0):
    #Fit line on points from sliding window in a list
    xs = np.array(xs)
    ys = np.array(ys)
    
    start_ind = point_ind-backward_window
    stop_ind = point_ind + forward_window+1

    if start_ind >= 0 and stop_ind <= len(xs):
        xpoints = xs[start_ind : stop_ind]
        ypoints = ys[start_ind : stop_ind]
    elif start_ind < 0:
        xpoints = xs[point_ind : stop_ind]
        ypoints = ys[point_ind : stop_ind]
    elif stop_ind > len(xs):
        xpoints = xs[start_ind : point_ind]
        ypoints = ys[start_ind : point_ind]

    #Compute linear fit
    coef = np.polyfit(xpoints,ypoints,deg=1)
    fit_fn = np.poly1d(coef) 
    
    # Get fitted line on point,vector form
    line_point = np.array([xs[point_ind],fit_fn(xs[point_ind])])
    if point_ind + 1 < len(xs):
        next_line_point = np.array([xs[point_ind+1],fit_fn(xs[point_ind+1])])
        line_vector = next_line_point - line_point
    else:
        next_line_point = np.array([xs[point_ind-1],fit_fn(xs[point_ind-1])])
        line_vector = line_point - next_line_point
    line_vector = line_vector/np.linalg.norm(line_vector)

    return line_point,line_vector

def line_XY_intersection(point, direction):
    """
    Finds intersection (x,y) between XY plane and a line.

    point: some point on the line (e.g. camera position)
    direction: some vector pointing along the line

    Assumes numpy arrays.
    """
    r = point[2]/direction[2]
    xy = point[0:2] - r*direction[0:2]
    return xy

def orient2d(a, b, c):
    """
    The Orient2D geometric predicate.

    c can be 2x1 or 2xN: c[0] is x values, c[1] is y values

    The output is a scalar number which is:
        > 0 if abc forms an angle in (0, pi), turning left,
        < 0 if abc forms an angle in (0, -pi), turning right,
        = 0 if abc forms an angle equal to 0 or pi, or is straight.

    Alternatively, it can be interpreted as:
        > 0 if c is to the left of the line ab,
        < 0 if c is to the right of the line ab,
        = 0 if c is on the line ab,
    in all cases seen from above, and along ab.

    The algorithm do not use exact arithmetics, and may fail within
    machine tolerances.
    """
    return (a[0]-c[0])*(b[1]-c[1]) - (a[1]-c[1])*(b[0]-c[0])