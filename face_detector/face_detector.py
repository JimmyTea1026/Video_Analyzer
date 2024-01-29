from face_detector.scrfd import SCRFD
import os
import dlib
import cv2

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))

class FaceDetector:
    def __init__(self, type) -> None:
        self.type = type
        self.prepare()
    
    def prepare(self):
        if self.type == 'scrfd':
            self.detector = SCRFD("/utlis/model/det_500m.onnx")
            self.detector.prepare(0)
        elif self.type == 'dlib':
            self.detector = dlib.get_frontal_face_detector()
        elif self.type == 'haar':
            self.detector= cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    def detect(self, frame):
        if self.type == 'scrfd':
            bboxes, kpss = self.detector.autodetect(frame, max_num=1)
            return bboxes
        elif self.type == 'dlib':
            rects = self.detector(frame, 1)
            if len(rects) == 0:
                return None
            else:
                rect = rects[0]
                x1 = rect.left()
                y1 = rect.top()
                x2 = rect.right()
                y2 = rect.bottom()
                
                return [x1, y1, x2, y2]
        elif self.type == 'haar':
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            bbox = self.detector.detectMultiScale(
                gray_frame, scaleFactor=1.1, minNeighbors=6, minSize=(80, 80)
            )
            if len(bbox) == 0:
                return None
            else:
                return bbox[0]
        