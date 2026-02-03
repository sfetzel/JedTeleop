import enum
from abc import ABC, abstractmethod
import threading
import time
from typing import Optional

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from jed_teleop.utils import calculate_rotation_matrix, ema_smooth


class GripperState(enum.Enum):
    Closed = -1.0
    Open = 1.0


class PoseEstimator(ABC):
    def __init__(self, stretch_factors: Optional[list] = None):
        self.latest_deltas = None
        self.current_pose: Optional[np.ndarray] = None
        self.last_normal = None
        self.thread = None
        self.stop_requested = False
        self.is_paused = False
        self.decay = 0.35
        self.zero_pos = np.zeros(3)
        self.is_gripper_closed = False
        self.stretch_factors = np.array(stretch_factors if stretch_factors is not None else [1.0, 1.0, 1.0])
        self.pos_lock = threading.Lock()
        self.normal_rot = None
        
    @abstractmethod
    def run(self):
        pass
    
    def get_deltas(self):
        with self.pos_lock:
            result = self.latest_deltas
            self.latest_deltas = None
        return result
        
    def start(self):
        if self.thread is None:
            self.thread = threading.Thread(target=self.run)
            self.thread.start()
            self.stop_requested = False
            
    def stop(self):
        if not self.thread is None:
            self.stop_requested = True
            self.thread.join()

    def set_pose_and_update_deltas(self, new_position):
        with self.pos_lock:
            if not self.is_paused:
                # if not paused and current position is not none, then calculate deltas.
                if self.current_pose is not None:
                    delta = new_position - self.current_pose

                    if self.latest_deltas is None:
                        self.latest_deltas = delta
                    else:
                        self.latest_deltas += delta

                    # the gripper value is absolute.
                    self.latest_deltas[-1] = new_position[-1]
            else:
                # reset deltas if paused.
                self.latest_deltas = None
            self.current_pose = new_position.copy()

    def set_smoothed_pose_and_update_deltas(self, new_pose):
        if self.current_pose is not None:
            # exponential moving average for all numbers but gripper state (last entry).
            new_pose[:-1] = ema_smooth(self.decay, new_pose, self.current_pose)[:-1]
        self.set_pose_and_update_deltas(new_pose)

    def process_key(self, key: int) -> bool:
        if key == ord('q'):
            return False

        if key == ord('c') and not self.last_normal is None:
            self.normal_rot = calculate_rotation_matrix(self.last_normal)

        if key == ord('z') and not self.current_pose is None:
            self.zero_pos = self.current_pose[:3]

        if key == ord('p'):
            self.is_paused = not self.is_paused
            print(f"Paused: {self.is_paused}")
        return True

    def shift_scale_location(self, location):
        new_location = location - self.zero_pos
        return new_location * self.stretch_factors

    def annotate_image(self, display_img):
        if self.is_paused:
            cv2.putText(display_img, f"paused",
                        (10, 10), cv2.FONT_HERSHEY_DUPLEX,
                        0.5, np.zeros(3), 1, cv2.LINE_AA)
        cv2.putText(display_img, f"Gripper: {'Closed' if self.is_gripper_closed else 'Open'}",
                    (10, 40), cv2.FONT_HERSHEY_DUPLEX,
                    0.5, np.zeros(3), 1, cv2.LINE_AA)

    def __del__(self):
        self.stop()

class MockEstimator(PoseEstimator):
    def run(self):
        while not self.stop_requested:
            time.sleep(0.1)
            self.latest_deltas = np.concatenate([-np.array([0.1, 0.05, 0.01]), np.array([0.1, 0, 0])])

class CircleEstimator(PoseEstimator):
    def __init__(self):
        super().__init__()
        self.position = np.array([0.15, 0.15, 0.15])
        self.rotation = R.from_euler('XYZ', [0, 0, 0.1])


    def run(self):
        while not self.stop_requested:
            new_location = self.rotation.apply(self.position)
            self.position = new_location
            new_position = np.concatenate([self.position, np.zeros(3), np.array([-1.0])])
            self.set_pose_and_update_deltas(new_position)
            time.sleep(0.1)

class RotatorEstimator(PoseEstimator):
    def __init__(self, rotation_delta, position = None):
        super().__init__()
        self.position = np.array([0.5, 0.5, 0.5]) if position is None else position
        self.rotation = np.zeros(3)
        self.rotation_delta = rotation_delta
        print(self.rotation_delta)
        self.current_position = np.zeros(7)
        self.current_position[:3] = self.position
        self.current_position[-1] = 1.0


    def run(self):
        while not self.stop_requested:
            self.latest_deltas = np.concatenate([np.zeros(3), self.rotation_delta, np.array([1.0])])
            self.current_position[:6] += np.concatenate([np.zeros(3), self.rotation_delta])
            time.sleep(0.1)
