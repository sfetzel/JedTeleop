import numpy as np


def mirror_point(a, b, c, d, x1, y1, z1):
    # code by https://www.geeksforgeeks.org/dsa/mirror-of-a-point-through-a-3-d-plane/
    k = (-a * x1 - b * y1 - c * z1 - d) / float((a * a + b * b + c * c))
    x2 = a * k + x1
    y2 = b * k + y1
    z2 = c * k + z1
    x3 = 2 * x2 - x1
    y3 = 2 * y2 - y1
    z3 = 2 * z2 - z1
    result = np.array([x3, y3, z3])
    return result / np.linalg.norm(result)

def calculate_plane_normal(points: np.ndarray) -> np.ndarray:
    A = np.zeros((points.shape[0], 3))
    A[:, 0] = points[:, 0].reshape(-1)
    A[:, 1] = points[:, 1].reshape(-1)
    A[:, 2] = 1

    b = points[:, 2].reshape(-1)  # np.zeros((A.shape[0], 1))
    solution, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    # z = a*x + b*y + c = solution[0]*x + solution[1]*y + solution[2]
    normal = [-solution[0], -solution[1], 1]
    unit_normal = normal / np.linalg.norm(normal)
    return unit_normal, [-solution[0], -solution[1], 1, -solution[2]]


def calc_normal(a, b, c):
    v1 = b-a
    v2 = c-a
    normal = np.cross(v1, v2)
    return normal / np.linalg.norm(normal)


def to_text(vector):
    return f"({vector[0]}, {vector[1]}, {vector[2]})"


def convert_hand_landmarks(hand_landmarks):
    # hand landmarks have coordinates in camera space, which is oriented along image (x,y) and depth z.
    # y is measured from top left of image.
    # 1-y is measured from bottom left of image.
    # z is measured from camera to point: larger value: farther away, smaller value: nearer.
    # for z we want: farther away: smaller value, near: large value.
    # transformation: x -> x, y -> (1-y) -> z, z -> y.
    hand_points = [[1-l.x, l.z, 1 - l.y] for l in hand_landmarks]
    #hand_points = [[l.x, l.y, l.z] for l in hand_landmarks]
    return hand_points

def calculate_normal(hand_points: list[list]) -> np.ndarray:
    points = np.array([
        hand_points[0], hand_points[5], hand_points[17],
    ])
    # direction normal is calculated from pinky to index finger, so the orientation is correct.
    normal = calc_normal(points[0, :], points[-1, :], points[1, :])
    normal[1] *= -1
    return normal
