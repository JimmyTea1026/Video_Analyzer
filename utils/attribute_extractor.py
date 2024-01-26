import cv2
import numpy as np
import matplotlib.pyplot as plt

class Attribute_extractor:
    def __init__(self) -> None:
        self.reset()
    
    def reset(self):
        self.result_list = {'gray':[], 'hsv':[], 'hist':[]}
    
    def inference(self, frame):
        self.frame = frame
        self.get_hsv()
        self.get_gray_hist()

    def get_gray_hist(self):
        gray_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray_frame], [0], None, [256], [0, 250])
        brightness = np.mean(hist)
        self.result_list['hist'].append(brightness)
        # cv2.imwrite("plot/gray_frame.jpg", gray_frame)
        plt.hist(gray_frame.ravel(), 256, [0, 250])
        plt.savefig("plot/hist2.jpg")

    def get_hsv(self):
        hsv_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
        v_channel = hsv_frame[:,:,2]
        brightness = int(v_channel.mean())
        self.result_list['hsv'].append(brightness)

    def get_result_list(self):
        return self.result_list