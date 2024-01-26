import cv2

class Video_loader:
    def __init__(self, angle=0) -> None:
        self.angle = angle
            
    def load_video(self, path):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            print("Error opening video stream or file")
            return False
        return True
    
    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if self.angle != 0:
            rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), self.angle, 1)
            frame = cv2.warpAffine(frame, rotation_matrix, (int(width), int(height)))
        # cv2.imwrite('test.jpg', rotated_frame)
        return frame
    
    def release(self):
        self.cap.release()