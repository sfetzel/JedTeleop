import numpy as np

from .pose_estimator import GripperState

def distance(pts, index1, index2):
    return np.linalg.norm(pts[index1, :] - pts[index2, :])

def detect_gripper_state(pts: np.ndarray, threshold = 0.7) -> GripperState:
    """
    Detects the gripper state using the given hand keypoints.
    :param pts: the hand keypoints in the shape (21, 3).
    :param threshold: if the finger circumference divided by the palm circumference is larger than this value, the gripper is open.
    :return: GripperState.
    """
    tip_distances = [
        distance(pts, 4, 8), # thumb to index
        distance(pts, 8, 12), # index to middle finger
        distance(pts, 12, 16), # middle to ring finger.
        distance(pts, 16, 20),  # middle to ring finger.
    ]
    palm_distances = [
        distance(pts, 0, 5),
        distance(pts, 5, 9),  # index to middle
        distance(pts, 9, 13),  # middle to ring
        distance(pts, 13, 17),  # ring to pinky.
        distance(pts, 17, 0),
    ]
    palm_circumference = np.array(palm_distances).sum()
    finger_circumference = np.array(tip_distances).sum()
    circumference_ratio = finger_circumference / palm_circumference

    new_is_gripper_closed = GripperState.Open if circumference_ratio > threshold else GripperState.Closed
    return new_is_gripper_closed