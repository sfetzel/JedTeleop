import threading

from jed_teleop.depth_estimator import DepthAnythingEstimator
from jed_teleop.sources.VideoSource import Frame, VideoSource
from jed_teleop.utils.opencv_capture import BufferlessCapture, DirectCapture

class OpenCvDepthEstSource(VideoSource):
    def __init__(self, capture_name, depth_decay = 0.25):
        super(OpenCvDepthEstSource, self).__init__()
        self.cap = DirectCapture(capture_name)
        self.depth_estimator = DepthAnythingEstimator(True, depth_decay)
        self.last_frame = None
        self.reader_thread.start()

    def _reader(self):
        while True:
            image = self.cap.get_frame()
            depth = self.depth_estimator.get_depth(image)
            self.last_frame = Frame(image, depth)
