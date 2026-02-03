import unittest
from jed_teleop.mediapipe_pose_estimator import MediaPipePoseEstimator
from os import path

from jed_teleop.sources.VideoSource import Frame

test_dir = path.join(path.dirname(__file__), "test_images")

class TestHandPoseEstimator(unittest.TestCase):
    def test_process_result(self):
        frame = Frame.from_file(path.join(test_dir, "color_0000.png"), path.join(test_dir, "depth_0000.png"))

        estimator = MediaPipePoseEstimator(None)
        estimator.process_frame(frame)

        self.assertIsNotNone(estimator.current_pose)
