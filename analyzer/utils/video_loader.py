import cv2

class Video_loader:
    def __init__(self, path, angle=0) -> None:
        self.angle = angle
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            print("Error opening video stream or file")
    
    def get_frame_num(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def get_video_fps(self):
        return int(self.cap.get(cv2.CAP_PROP_FPS))

    def get_video_length(self):
        return round(self.get_frame_num() / self.get_video_fps(), 1)
    
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