import numpy as np


def calculate_rotation_matrix(unit_normal: np.ndarray, rotate_into: np.ndarray = None) -> np.ndarray:
    if rotate_into is None:
        rotate_into = np.array([0,0,1])
    v = np.cross(unit_normal, rotate_into)
    c = np.dot(unit_normal, rotate_into)

    vx = np.array([[0.0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + vx + np.dot(vx, vx) * 1/(1+c)
    return R

def cart2sph(vec):
    x,y,z = vec
    XsqPlusYsq = x**2 + y**2
    r = np.sqrt(XsqPlusYsq + z**2)               # r
    polar = np.arccos(z/r) /np.pi * 180     # theta
    az = np.arctan2(y,x) /np.pi*180                           # phi
    return r, polar, az

def to_image_indices(relative_coordinate, length: int) -> int:
    return int(max(0, min(relative_coordinate, 1.0)) * (length - 1))

def to_text(vector):
    return f"({vector[0]}, {vector[1]}, {vector[2]})"

def ema_smooth(decay: float, new_vector: np.ndarray, old_vector: np.ndarray):
    """
    Applies exponential moving average filter to new vector
    :param decay: the decay factor for the new vector.
    :param new_vector: the new vector.
    :param old_vector: the old/current vector.
    :return: decay * new_vector + (1-decay) * old_vector
    """
    return decay * new_vector + (1-decay) * old_vector

def combine_points_width_depth(points_2d: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """
    Extends the 2d point array width the depth from the provided depth image.
    :param points_2d: 2d point array in the format (x,y) in relative coordinates in the image frame.
    :param depth: depth image with shape (height, width).
    :return: extended 2d point array
    """
    depth_height, depth_width = depth.shape
    depth_values = []
    for point in points_2d:
        depth_y = to_image_indices(point[1], depth_height)
        depth_x = to_image_indices(point[0], depth_width)
        depth_values.append(depth[depth_y, depth_x])
    depth_array = np.array([depth_values])
    return np.hstack([points_2d, depth_array.T])