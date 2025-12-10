import os

import cv2
import numpy as np
import pytest

from jed_teleop import GripperState
from jed_teleop.grip_detector import detect_gripper_state
from jed_teleop.hands_detection.mp_hands import VisionRunningMode, MediaPipeHandPose

open_folder = "test/open"
closed_folder = "test/close"
open_images = [os.path.join(open_folder, image) for image in os.listdir(open_folder)]
close_images = [os.path.join(closed_folder, image) for image in os.listdir(closed_folder)]

detector = MediaPipeHandPose(min_hand_detected_confidence=0.3, min_hand_presence_confidence=0.3,
                             running_mode=VisionRunningMode.IMAGE)
def get_hand_landmarks(image):
    img = cv2.imread(image)
    landmarks, handedness = detector.detect(img)
    return img, landmarks, handedness

def check_image(image, expected_state):
    img, landmarks, handedness = get_hand_landmarks(image)
    pts = np.array([[l.x, l.y, l.z] for l in landmarks])
    actual_state = detect_gripper_state(pts)
    assert actual_state == expected_state

@pytest.mark.parametrize("image", open_images)
def test_detect_grip_should_be_open(image):
    check_image(image, GripperState.Open)

@pytest.mark.parametrize("image", close_images)
def test_detect_grip_should_be_closed(image):
    check_image(image, GripperState.Closed)