import cv2
import numpy as np
import pytest

import os

from jed_teleop.hands_detection.mp_hands import MediaPipeHandPose, VisionRunningMode
from jed_teleop.orientation import calculate_normal, convert_hand_landmarks
from jed_teleop.utils.utils import cart2sph, to_text

detector = MediaPipeHandPose(min_hand_detected_confidence=0.3, min_hand_presence_confidence=0.3,
                             running_mode=VisionRunningMode.IMAGE)
test_folder = "test_images"
images = os.listdir(test_folder)

def get_hand_landmarks(image):
    img = cv2.imread(os.path.join(test_folder, image))
    img = cv2.flip(img, 1)
    landmarks, handedness = detector.detect(img)
    return img, landmarks, handedness

def create_debug_image(image, hand_landmarks, handedness, filename):
    annotated_image = MediaPipeHandPose.annotate_image(image, hand_landmarks, handedness)
    cv2.imwrite(os.path.join(test_folder, filename), annotated_image)

z_dir = np.array([0,0,1])
y_dir = np.array([0,1,0])
x_dir = np.array([1,0,0])

test_data = [("hand-posz.png", z_dir),
             ("hand-posz2.png", z_dir),
             ("hand-posz3.png", z_dir),
             ("hand-posz4.png", z_dir),
             ("hand-posz5.png", z_dir),
             ("hand-negy.png", -y_dir),
             ("hand-negy2.png", -y_dir),
             ("hand-posy.png", y_dir),
             ("hand-posx.png", x_dir),]

@pytest.mark.parametrize("image,expected_normal", test_data)
def test_direction(image, expected_normal):
    img, landmarks, handedness = get_hand_landmarks(image)
    points = convert_hand_landmarks(landmarks)
    normal = calculate_normal(points)
    distance = np.linalg.norm(expected_normal - normal)

    create_debug_image(img, landmarks, handedness, f"{image}-debug.png")
    assert distance < 8e-1, f"Distance is too large: {distance}, normal: {to_text(normal)}"

@pytest.mark.parametrize("image,expected_polar,expected_azimuth", [
    ("hand-negx-45.png", 45, 180),
    ("hand-negx2-45.png", 45, 180),
    ("hand-posx-45.png", 45, 0),
])
def test_angles(image, expected_polar, expected_azimuth):
    img, landmarks, handedness = get_hand_landmarks(image)
    points = convert_hand_landmarks(landmarks)
    normal = calculate_normal(points)
    _, polar, az = cart2sph(normal)
    assert abs(polar - expected_polar) < 20, f"Polar is {polar}, but expected is {expected_polar}"
    assert abs(az - expected_azimuth) < 50, f"Azimuth is {az}, but expected is {expected_azimuth}"

