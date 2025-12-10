import enum
from abc import ABC, abstractmethod
import threading
import time
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

class GripperState(enum.Enum):
    Closed = -1.0
    Open = 1.0


class PoseEstimator(ABC):
    def __init__(self):
        self.latest_deltas = None
        self.current_position: Optional[np.ndarray] = None
        self.thread = None
        self.stop_requested = False
        self.is_paused = False
        
    @abstractmethod
    def run(self):
        pass
    
    def get_deltas(self):
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

    def set_position_and_update_deltas(self, new_position):
        if not self.is_paused:
            # if not paused and current position is not none, then calculate deltas.
            if self.current_position is not None:
                delta = new_position - self.current_position

                if self.latest_deltas is None:
                    self.latest_deltas = delta
                else:
                    self.latest_deltas += delta

                # the gripper value is absolute.
                self.latest_deltas[-1] = new_position[-1]
        else:
            # reset deltas if paused.
            self.latest_deltas = None
        self.current_position = new_position.copy()

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
            self.set_position_and_update_deltas(new_position)
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
