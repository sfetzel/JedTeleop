import cv2
import numpy as np

from jed_teleop.hands_detection.hands import Handedness
from jed_teleop.hands_detection.mp_hands import MediaPipeHandPose, VisionRunningMode

test_image = "test_images/hand-posy.png"

def test_detect():
    detector = MediaPipeHandPose(running_mode=VisionRunningMode.IMAGE)
    image = cv2.imread(test_image)
    assert image is not None
    result = detector.detect(image)
    assert result is not None, "Result should not be None"
    landmarks, handedness = result
    assert handedness == Handedness.RIGHT, "Hand on image is a right hand."
    assert len(landmarks) == 21


def test_detect_no_hand():
    detector = MediaPipeHandPose(running_mode=VisionRunningMode.IMAGE)
    image = np.zeros((512, 512, 3), np.uint8)
    assert image is not None
    result = detector.detect(image)
    assert result is None, "Result should be None"