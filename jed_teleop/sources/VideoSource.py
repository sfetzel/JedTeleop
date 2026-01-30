import threading
from abc import ABC, abstractmethod
from typing import Optional


class Frame:
    def __init__(self, rgb_image, depth_image):
        self.rgb_image = rgb_image
        self.depth = depth_image

class VideoSource(ABC):
    def __init__(self):
        self.last_frame : Optional[Frame] = None
        self.reader_thread = threading.Thread(target=self._reader)
        self.reader_thread.daemon = True

    @abstractmethod
    def _reader(self):
        pass