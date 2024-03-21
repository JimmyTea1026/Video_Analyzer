import cv2
import numpy as np
import matplotlib.pyplot as plt

class Brightness_detector:
    def __init__(self) -> None:
        self.brightness = 0
        self.pos = (0, 0)

    def inference(self, img, rect, mode='hsv'):
        '''
        Input the cropped face image and mode, return the brightness of the image
        '''
        image = img.copy()
        x1, y1, x2, y2 = rect
        self.face = image[y1:y2, x1:x2]
        self.brightness = None
        if mode == 'hsv':
            self.brightness = self.get_hsv()
        elif mode == 'gray':
            self.brightness = self.get_gray_hist()
            
        self.pos = (x1+2, y2+2)
        cv2.putText(image, f"self.brightness : {self.brightness}", self.pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return self.brightness, image

    def get_range(self, masked_frame):
        # Calculate the pixel value distribution, mean, and standard deviation of the image, but exclude the white part
        gray_frame = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
        face_indices = np.where(gray_frame != 255)
        face_pixels = gray_frame[face_indices]
        mean_value = np.mean(face_pixels, axis=-1)
        std_value = np.std(face_pixels, axis=-1)
        
        return [mean_value, std_value]

    def draw_image(self, img):
        image = img.copy()
        cv2.putText(image, f"self.brightness : {self.brightness}", self.pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return image
    
    def get_gray_hist(self):
        gray_frame = cv2.cvtColor(self.face, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray_frame], [0], None, [256], [0, 250])
        self.brightness = np.mean(hist)
        return self.brightness

    def get_hsv(self):
        hsv_frame = cv2.cvtColor(self.face, cv2.COLOR_BGR2HSV)
        v_channel = hsv_frame[:,:,2]
        self.brightness = int(v_channel.mean())
        return self.brightness