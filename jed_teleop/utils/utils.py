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