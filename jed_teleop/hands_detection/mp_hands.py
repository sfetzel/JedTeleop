import time
from typing import Tuple, Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import os

from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark

from jed_teleop.hands_detection.hands import HandPoseDetector, Handedness

VisionRunningMode = mp.tasks.vision.RunningMode
model_path_full = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green
CAPTION_TEXT_COLOR = (0, 0, 0) # vibrant green

class MediaPipeHandPose(HandPoseDetector):
    """
    Proxy class for mediapipe hand pose detection.
    Code is based on examples from Google:
    https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python
    https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb#scrollTo=_JVO3rvPD4RN&uniqifier=1
    """
    def __init__(self, model_path: str = model_path_full, min_hand_detected_confidence=0.5,
                 min_hand_presence_confidence=0.5, running_mode = VisionRunningMode.VIDEO):
        # Create an HandLandmarker object
        options = vision.HandLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=model_path),
                                            running_mode=running_mode,
                                            num_hands=1,
                                            min_hand_detection_confidence=min_hand_detected_confidence,
                                            min_hand_presence_confidence=min_hand_presence_confidence,
                                            min_tracking_confidence=0.5)
        self.running_mode = running_mode
        self.detector = vision.HandLandmarker.create_from_options(options)

    def detect(self, img: np.ndarray)-> Optional[Tuple[list[NormalizedLandmark], Handedness]]:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)

        timestamp = int(round(time.time() * 1000))
        if self.running_mode == VisionRunningMode.VIDEO:
            hand_landmarker_result = self.detector.detect_for_video(mp_image, timestamp)
        else:
            hand_landmarker_result = self.detector.detect(mp_image)

        if len(hand_landmarker_result.hand_landmarks) == 0:
            return None

        handedness = Handedness.LEFT if hand_landmarker_result.handedness[0][0].category_name == "Left" else Handedness.RIGHT
        return hand_landmarker_result.hand_landmarks[0], handedness

    @classmethod
    def annotate_image(cls, image: np.ndarray, hand_landmarks: list[NormalizedLandmark], handedness: Handedness) -> np.ndarray:
        annotated_image = image.copy()
        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style())

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Draw handedness (left or right hand) on the image.
        cv2.putText(annotated_image, f"{handedness.value}",
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

        for i, landmark in enumerate(hand_landmarks):
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            cv2.putText(annotated_image, f"{i}",
                        (x, y), cv2.FONT_HERSHEY_DUPLEX,
                        0.5, CAPTION_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

        return annotated_image