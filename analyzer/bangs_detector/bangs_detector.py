import cv2
import numpy as np
import os

class Bangs_detector:
    def __init__(self) -> None:
        pass
    
    def inference(self, img, rect):
        '''
        Input the cropped face image, return the proportion of skin and merged image
        '''
        x1, y1, x2, y2 = rect
        face_image = img[y1:y2, x1:x2]
        skin_mask, hair_mask = self.get_masks(face_image)

        # 計算頭髮區域的像素數量
        hair_pixels = cv2.countNonZero(hair_mask) 
        all_pixels = (x2-x1) * (y2-y1)
        hair_proportion = round(hair_pixels / all_pixels, 2)
   
        return hair_proportion, None
    
    def get_masks(self, face_image):
        hsv_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2HSV)
        ycrcb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2YCrCb)     
        # 定義膚色的YCbCr範圍
        lower_skin = np.array([0, 140, 77], dtype=np.uint8)
        upper_skin = np.array([255, 173, 123], dtype=np.uint8)  
        # YCbCr, Cr 為紅色偏移量，只要調整這個就好

        # 定義頭髮的HSV範圍
        lower_hair = np.array([0, 0, 0], dtype=np.uint8)
        upper_hair = np.array([179, 255, 70], dtype=np.uint8)

        # 創建膚色和頭髮的遮罩
        skin_mask = cv2.inRange(ycrcb_image, lower_skin, upper_skin)
        hair_mask = cv2.inRange(hsv_image, lower_hair, upper_hair)
        return skin_mask, hair_mask
    
    def get_masked_frame(self, frame, rect):
        x1, y1, x2, y2 = rect
        face_image = frame[y1:y2, x1:x2]
        skin_mask, hair_mask = self.get_masks(face_image)
        skin_result = self.mask_frame(face_image, skin_mask)
        hair_result = self.mask_frame(face_image, hair_mask)
        merge_result = self.merge_horizontally(face_image, skin_result, hair_result)
        
        return skin_result, merge_result

    def mask_frame(self, frame, mask):
        result = cv2.bitwise_and(frame, frame, mask=mask)
        gray_result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        black_mask = cv2.threshold(gray_result, 1, 255, cv2.THRESH_BINARY_INV)[1]
        result[black_mask == 255] = [255, 255, 255]
        return result

    def merge_horizontally(self, img, skin_result, hair_result):
        space = np.zeros((img.shape[0], 20, 3), dtype=np.uint8)
        merged_image = cv2.hconcat([img, space, skin_result, space, hair_result])
        return merged_image
    
    def draw_image(self, img):
        if len(self.cur_mask) == 0:
            return img
        image = img.copy()
        skin_result, hair_result = self.cur_mask
        cv2.putText(image, f"skin_proportion : {self.skin_proportion}", (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        skin_y = skin_result.shape[0]
        skin_x = skin_result.shape[1]
        hair_y = hair_result.shape[0]
        hair_x = hair_result.shape[1]
        bia = 40
        image[bia:bia+skin_y, 0:skin_x] = skin_result
        image[bia:bia+hair_y, skin_x:skin_x+hair_x] = hair_result
        
        return image
