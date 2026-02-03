import cv2
import time

import numpy as np
from scipy.spatial.transform import Rotation as R

from jed_teleop.keypoint_based_estimator import KeypointBasedEstimator
from jed_teleop.sources.VideoSource import VideoSource, Frame
from .hands_detection.mp_hands import MediaPipeHandPose, VisionRunningMode
from .orientation import convert_hand_landmarks, calculate_normal

from jed_teleop.utils import calculate_rotation_matrix, combine_points_width_depth, ema_smooth


class MediaPipePoseEstimator(KeypointBasedEstimator):

    def __init__(self, source: VideoSource, stretch_factors: list = None) -> None:
        super(MediaPipePoseEstimator, self).__init__(source, stretch_factors)
        self.detector = MediaPipeHandPose(running_mode=VisionRunningMode.VIDEO)
        self.zero_pos = np.array([0.5, 0.5, 0.5])
        self.horizontal_flip = True

    def process_result(self, detection_result, rgb_image: np.ndarray, depth: np.ndarray):
        hand_landmarks, handedness = detection_result

        # points in camera space (origin is at left bottom).
        points_camera = convert_hand_landmarks(hand_landmarks)
        normal = calculate_normal(points_camera)
        self.last_normal = normal

        if not self.normal_rot is None:
            normal = self.normal_rot @ normal

        rotation = R.from_matrix(calculate_rotation_matrix(normal))
        new_rotation = rotation.as_euler("xyz")

        points_image = np.array([[l.x, l.y, l.z] for l in hand_landmarks])
        self.set_gripper_state(points_image)

        # naive lifting without perspective projection correction, because that would be too noisy.
        points_3d = combine_points_width_depth(points_image[:, :2], depth)
        points_palm = np.array([points_3d[0, :], points_3d[5, :], points_3d[9, :], points_3d[13, :], points_3d[17, :]])
        points_palm = points_palm[points_palm[:, 2] != 0.0]
        if len(points_palm) > 0:
            palm_center = np.mean(points_palm, axis=0) # average across all points.
        else:
            # use last position if depth is unavailable.
            palm_center = self.current_pose[:3]

        new_location = self.shift_scale_location(np.array([palm_center[0], palm_center[2], (1 - palm_center[1]) ]))

        gripper_value = self.get_gripper_state_int()

        new_pose = np.concatenate([new_location, new_rotation, np.array([gripper_value])])
        self.set_smoothed_pose_and_update_deltas(new_pose)

    def process_frame(self, frame: Frame):
        img, depth = frame.rgb_image, frame.depth

        if self.horizontal_flip:
            img = cv2.flip(img, 1)
            depth = cv2.flip(depth, 1)
        hand_landmarker_result = self.detector.detect(img)
        display_img = np.dstack((depth, depth, depth))

        if hand_landmarker_result is not None:
            self.process_result(hand_landmarker_result, img, depth)
            hand_landmarks, handedness = hand_landmarker_result
            display_img = MediaPipeHandPose.annotate_image(display_img, hand_landmarks, handedness)
        return display_img


if __name__ == "__main__":
    m = MediaPipePoseEstimator(0)
    m.run()
