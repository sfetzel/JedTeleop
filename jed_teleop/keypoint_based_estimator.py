import time
from abc import abstractmethod

import cv2

from .pose_estimator import PoseEstimator, GripperState
from .grip_detector import detect_gripper_state
from .sources.VideoSource import VideoSource


class KeypointBasedEstimator(PoseEstimator):
    def __init__(self, source:VideoSource, stretch_factors: list = None) -> None:
        super().__init__(stretch_factors)
        self.source = source
        self.finger_distance_threshold = 0.07

    def set_gripper_state(self, keypoints):
        self.is_gripper_closed = detect_gripper_state(keypoints) == GripperState.Closed

    def get_gripper_state_int(self):
        return GripperState.Closed.value if self.is_gripper_closed else GripperState.Open.value

    @abstractmethod
    def process_frame(self, frame):
        pass

    def run(self):
        while not self.stop_requested:
            last_frame = self.source.last_frame

            if last_frame is None:
                time.sleep(0.1)
                continue

            display_img = self.process_frame(last_frame)

            self.annotate_image(display_img)
            cv2.imshow('Hand detection', display_img)

            if not self.process_key(cv2.waitKey(1) & 0xFF):
                break