import threading
from abc import ABC, abstractmethod
from typing import Optional
import os

import cv2


class Frame:
    def __init__(self, rgb_image, depth_image):
        self.rgb_image = rgb_image
        self.depth = depth_image

    @staticmethod
    def from_file(img_path, depth_path):
        assert(os.path.exists(img_path))
        assert(os.path.exists(depth_path))

        img = cv2.imread(img_path)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        return Frame(img, depth)

class VideoSource(ABC):
    def __init__(self):
        self.last_frame : Optional[Frame] = None
        self.reader_thread = threading.Thread(target=self._reader)
        self.reader_thread.daemon = True
        self.stop_requested = False


    @abstractmethod
    def _reader(self):
        pass

    def __del__(self):
        self.stop_requested = True
        self.reader_thread.join()