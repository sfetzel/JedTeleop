import cv2
import numpy as np
import plotly.graph_objects as go
import pytest

import os

from mediapipe.tasks.python.components.containers import NormalizedLandmark

from jed_teleop.hands_detection.mp_hands import MediaPipeHandPose, VisionRunningMode
from jed_teleop.orientation import calculate_normal, convert_hand_landmarks
from jed_teleop.utils.utils import cart2sph, to_text

detector = MediaPipeHandPose(min_hand_detected_confidence=0.3, min_hand_presence_confidence=0.3,
                             running_mode=VisionRunningMode.IMAGE)
test_folder = "test/test_images"
images = os.listdir(test_folder)

def get_hand_landmarks(image):
    img = cv2.imread(os.path.join(test_folder, image))
    img = cv2.flip(img, 1)
    landmarks, handedness = detector.detect(img)
    return img, landmarks, handedness

def create_debug_image(image, hand_landmarks, handedness, filename):
    annotated_image = MediaPipeHandPose.annotate_image(image, hand_landmarks, handedness)
    cv2.imwrite(os.path.join(test_folder, filename), annotated_image)

def create_3d_debug_image(image, hand_landmarks: list[NormalizedLandmark], filename: str):
    x = [l.x for l in hand_landmarks]
    y = [l.y for l in hand_landmarks]
    z = [l.z for l in hand_landmarks]

    fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers')])
    fig.write_html(os.path.join(test_folder, filename))

z_dir = np.array([0,0,1])
y_dir = np.array([0,1,0])
x_dir = np.array([1,0,0])

test_data = [("hand-posz.png", z_dir),
             ("hand-posz2.png", z_dir),
             ("hand-posz3.png", z_dir),
             ("hand-posz4.png", z_dir),
             ("hand-posz5.png", z_dir),
             ("hand-posz6.png", z_dir),
             ("hand-posz7.png", z_dir),
             ("hand-posz8.png", z_dir),
             ("hand-posz9.png", z_dir),
             ("hand-posz10.png", z_dir),
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
    create_3d_debug_image(img, landmarks, f"{image}-debug.html")
    assert distance < 9e-1, f"Distance is too large: {distance}, normal: {to_text(normal)}; actual vector: {normal}"

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

