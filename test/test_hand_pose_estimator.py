import unittest
from jed_teleop.hand_pose_estimator import HandPoseEstimator
from os import path

from jed_teleop.sources.VideoSource import Frame

test_dir = path.join(path.dirname(__file__), "test_images")

class TestHandPoseEstimator(unittest.TestCase):
    def test_process_result(self):
        frame = Frame.from_file(path.join(test_dir, "color_0000.png"), path.join(test_dir, "depth_0000.png"))

        estimator = HandPoseEstimator(None)
        estimator.process_frame(frame)

        self.assertIsNotNone(estimator.current_position)
