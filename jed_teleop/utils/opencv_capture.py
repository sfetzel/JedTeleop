import cv2
import time
import threading

class BufferlessCapture:
    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.cap.set(cv2.CAP_PROP_FPS, 20)
        if not self.cap.isOpened():
            raise Exception("Could not open video.")
        self.last_frame = None
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                raise Exception("Could not read frame.")
            self.last_frame = frame

    def get_frame(self):
        return self.last_frame

class DirectCapture:
    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.cap.set(cv2.CAP_PROP_FPS, 20)
        if not self.cap.isOpened():
            raise Exception("Could not open video.")


    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            raise Exception("Could not read frame.")
        return frame
        
if __name__ == '__main__':
    reader = BufferlessCapture(0)

    while True:
        image = reader.last_frame
        if not image is None:
            cv2.imshow('Hand detection', image)
            cv2.waitKey(1)
            time.sleep(2)
