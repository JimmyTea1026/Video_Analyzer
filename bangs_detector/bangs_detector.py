import cv2
import numpy as np
import os

class Bangs_detector:
    def __init__(self) -> None:
        self.cur_mask = []
    
    def inference(self, img, rect):
        '''
        Input the cropped face image, return the proportion of skin and merged image
        '''
        self.skin_proportion = 0
        self.cur_mask.clear()
        
        x1, y1, x2, y2 = rect
        face_image = img[y1:y2, x1:x2]
        
        hsv_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2HSV)
        ycrcb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2YCrCb)
        # 定義膚色的YCbCr範圍
        lower_skin = np.array([0, 140, 77], dtype=np.uint8)
        upper_skin = np.array([255, 173, 118], dtype=np.uint8)  
        # YCbCr, Cr 為紅色偏移量，只要調整這個就好

        # 定義頭髮的HSV範圍
        lower_hair = np.array([0, 0, 0], dtype=np.uint8)
        upper_hair = np.array([179, 255, 70], dtype=np.uint8)

        # 創建膚色和頭髮的遮罩
        skin_mask = cv2.inRange(ycrcb_image, lower_skin, upper_skin)
        hair_mask = cv2.inRange(hsv_image, lower_hair, upper_hair)

        # 將遮罩範圍區域畫在原始圖像上
        skin_result = cv2.bitwise_and(face_image, face_image, mask=skin_mask)
        gray_sm = cv2.cvtColor(skin_result, cv2.COLOR_BGR2GRAY)
        black_mask = cv2.threshold(gray_sm, 1, 255, cv2.THRESH_BINARY_INV)[1]
        skin_result[black_mask == 255] = [255, 255, 255]
        
        hair_result = cv2.bitwise_and(face_image, face_image, mask=hair_mask)
        gray_sm = cv2.cvtColor(hair_result, cv2.COLOR_BGR2GRAY)
        black_mask = cv2.threshold(gray_sm, 1, 255, cv2.THRESH_BINARY_INV)[1]
        hair_result[black_mask == 255] = [255, 255, 255]
        self.cur_mask = [skin_result, hair_result]
        
        # Horizontally concatenate the images
        merged_image = self.merge_horizontally(face_image, skin_result, hair_result)

        skin_pixels = cv2.countNonZero(skin_mask) # 計算膚色區域的像素數量
        hair_pixels = cv2.countNonZero(hair_mask) # 計算頭髮區域的像素數量
        self.skin_proportion = round(skin_pixels / (skin_pixels + hair_pixels), 2)
   
        return self.skin_proportion, merged_image

    def merge_horizontally(self, img, skin_result, hair_result):
        space = np.zeros((img.shape[0], 20, 3), dtype=np.uint8)
        merged_image = cv2.hconcat([img, space, skin_result, space, hair_result])
        return merged_image
    
    def draw_image(self, img):
        if len(self.cur_mask) == 0:
            return img
        skin_result, hair_result = self.cur_mask
        cv2.putText(img, f"skin_proportion : {self.skin_proportion}", (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        img[40:40+skin_result.shape[0], 0:skin_result.shape[1]] = skin_result
        img[40+skin_result.shape[0]:40+skin_result.shape[0]+hair_result.shape[0], 0:hair_result.shape[1]] = hair_result
        
        return img
