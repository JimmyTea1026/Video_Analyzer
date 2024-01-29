import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np
import math
import cv2
import dlib
import matplotlib.pyplot as plt


class FaceeRotationAngleDetector:
    def __init__(self) -> None:
        self.ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        self.MODEL_PATH = os.path.join(self.ROOT_DIR, "model")
        self.SHAPE_MODEL_PATH = os.path.join(self.MODEL_PATH, "shape_predictor_68_face_landmarks.dat")
        self.face_detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(self.SHAPE_MODEL_PATH)
        self.cur_data = []
        
    def inference(self, img, rect):
        self.cur_data.clear()
        face_points = self.get_landmarks(img, rect)
        
        model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-225.0, 170.0, -135.0),     # Left eye left corner
                            (225.0, 170.0, -135.0),      # Right eye right corne
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner                         
                        ])
        size = img.shape
        focal_length = size[1]
        center = (size[1]/2, size[0]/2)
        camera_matrix = np.array(
                         [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype = "double"
                         )
        dist_coeffs = np.zeros((4,1))

        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, face_points, camera_matrix
                                                              , dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        if not success:
            return None, img
        self.cur_data = [rotation_vector, translation_vector, camera_matrix, dist_coeffs, face_points]
        yaw, pitch, roll = self.vector_to_euler(rotation_vector, translation_vector)
        drawed_image = self.draw_image(img.copy())

        return [yaw, pitch, roll], drawed_image
        
    def shape_to_np(self, shape, dtype="int"):
        # initialize the list of (x, y)-coordinates
        coords = np.zeros((68, 2), dtype=dtype)
    
        # loop over the 68 facial landmarks and convert them
        # to a 2-tuple of (x, y)-coordinates
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
    
        # return the list of (x, y)-coordinates
        return coords

    # 偵測單一人臉的臉部特徵(假設圖像中只有一個人)
    def get_landmarks(self, img, rect):
        rectangle = dlib.rectangle(rect[0], rect[1], rect[2], rect[3])
        shape = self.shape_predictor(img, rectangle)
        coords = self.shape_to_np(shape, dtype="int")
        
        nose_tip = coords[33:34] # 鼻尖 Nose tip: 34
        chin = coords[8:9] # 下巴 Chin: 9
        left_eye_corner = coords[36:37] # 左眼左角 Left eye left corner: 37
        right_eye_corner = coords[45:46] # 右眼右角 Right eye right corner: 46
        left_mouth_corner = coords[48:49] # 嘴巴左角 Left Mouth corner: 49
        right_mouth_corner = coords[54:55] # 嘴巴右角 Right Mouth corner: 55
        face_points = np.concatenate((nose_tip, chin, left_eye_corner, 
                                      right_eye_corner, left_mouth_corner, 
                                      right_mouth_corner)).astype(np.double)

        return face_points

    def vector_to_euler(self, rotation_vector, translation_vector):
        # 計算歐拉角
        rvec_matrix = cv2.Rodrigues(rotation_vector)[0]
        proj_matrix = np.hstack((rvec_matrix, translation_vector))
        eulerAngles = -cv2.decomposeProjectionMatrix(proj_matrix)[6]

        yaw   = eulerAngles[1]
        pitch = eulerAngles[0]
        roll  = eulerAngles[2]

        if pitch > 0:
            pitch = 180 - pitch
        elif pitch < 0:
            pitch = -180 - pitch
            yaw = -yaw
        
        return yaw, pitch, roll
    
    def draw_image(self, img):
        if len(self.cur_data) == 0:
            return img

        rotation_vector, translation_vector, camera_matrix, dist_coeffs, face_points = self.cur_data

        # 投射一個3D的點 (100.0, 0, 0)到2D圖像的座標上
        (x_end_point2D, jacobian) = cv2.projectPoints(np.array([(100.0, 0.0, 0.0)]), rotation_vector
                                                        , translation_vector, camera_matrix, dist_coeffs)

        # 投射一個3D的點 (0, 100.0, 0)到2D圖像的座標上
        (y_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 100.0, 0.0)]), rotation_vector
                                                        , translation_vector, camera_matrix, dist_coeffs)

        # 投射一個3D的點 (0, 0, 100.0)到2D圖像的座標上
        (z_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 100.0)]), rotation_vector
                                                , translation_vector, camera_matrix, dist_coeffs)


        # 以 Nose tip為中心點畫出x, y, z的軸線
        p_nose = (int(face_points[0][0]), int(face_points[0][1]))

        p_x = (int(x_end_point2D[0][0][0]), int(x_end_point2D[0][0][1]))

        p_y = (int(y_end_point2D[0][0][0]), int(y_end_point2D[0][0][1]))

        p_z = (int(z_end_point2D[0][0][0]), int(z_end_point2D[0][0][1]))

        cv2.line(img, p_nose, p_x, (0,0,255), 3)  # X軸 (紅色)
        cv2.line(img, p_nose, p_y, (0,255,0), 3)  # Y軸 (綠色)
        cv2.line(img, p_nose, p_z, (255,0,0), 3)  # Z軸 (藍色)

        # 把6個基準點標註出來
        for p in face_points:
            cv2.circle(img, (int(p[0]), int(p[1])), 3, (255,255,255), -1)

        return img
