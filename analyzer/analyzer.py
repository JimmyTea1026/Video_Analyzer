from concurrent.futures import ThreadPoolExecutor
import os
import pickle
from .utils.video_loader import Video_loader
from .brightness_detector.brightness_detector import Brightness_detector
from .bangs_detector.bangs_detector import Bangs_detector
from .angle_detector.angle_detector import FaceeRotationAngleDetector
from .mask_detector.mask_detector import FaceMaskDetector
# from .face_detector.face_detector import FaceDetector
import cv2

class Analyzer:
    def __init__(self) -> None:
        #---------------------------------------#
        BRIGHTNESS_DETECTOR = Brightness_detector()
        ROTATION_DETECTOR = FaceeRotationAngleDetector()
        MASK_DETECTOR = FaceMaskDetector()
        BANGS_DETECTOR = Bangs_detector()
        self.detectors = {'mask': MASK_DETECTOR, 'rotate': ROTATION_DETECTOR, 
                          'bangs': BANGS_DETECTOR, 'brightness': BRIGHTNESS_DETECTOR}
        #---------------------------------------#
    
    def inference(self, frame, rect=None):
        '''
        Input the origin image and cropped face rect, return the result of each detector.
        mask : True(Mask) / False(No mask)
        bangs : float(0~1) (The proportion of skin)
        rotate : [yaw, pitch, roll]
        brightness : int(0~255)
        '''
        result_dict = {'mask': None, 'rotate': None, 'bangs': None, 'attribute': None}

        if rect is None:
            raise ValueError("No face detected")
            # rect = self.FACE_DETECTOR.detect(frame)
            # if rect is None:
            #     return "No face detected"
    
        for key, detector in self.detectors.items():
            # detector.inference return result and image
            result, _ = detector.inference(frame, rect)
            result_dict[key] = result
        
        mask_result, bangs_result, rotate_result, brightness_result = self.result_process(result_dict)    
        return mask_result, bangs_result, rotate_result, brightness_result 
    
    def result_process(self, result_dict):
        mask_result = result_dict['mask']
        bangs_result = result_dict['bangs']
        rotate_result = result_dict['rotate']
        brightness_result = result_dict['brightness']
        return mask_result, bangs_result, rotate_result, brightness_result
    
    def inference_mask(self, frame):
        mask_result, _ = self.detectors['mask'].inference(frame)
        return mask_result  #return true or false
'''
    def get_video_attribute(self, video_path):
        # for test
        self.FACE_DETECTOR = FaceDetector('scrfd')
        video_name = video_path.split('/')[-1].split('.')[0]
        VIDEO_LOADER = Video_loader(video_path, 0)

        frame_count = 0
        face_list = []
        conbined_image_list = []
        result_dict = {'name': video_name, 'mask': [], 'rotate': [], 'bangs': [], 'brightness': []}
        image_dict = {'mask': [], 'rotate': [], 'bangs': [], 'brightness': []}
        fps = VIDEO_LOADER.get_video_fps()
        start_time = 1.5
        while True:
            frame = VIDEO_LOADER.get_frame()
            if frame is None:
                break
        
            frame_count += 1
            current_time = round(frame_count/fps, 1)
            if current_time <= start_time: 
                continue
            
            conbined_image = frame.copy()
            # Face detection
            rect = self.FACE_DETECTOR.detect(frame)
            if rect is None: 
                for _, item in result_dict.items():
                    if item == list:
                        item.append(None)
            else:
                x1, y1, x2, y2 = rect
                face = frame[y1:y2, x1:x2]
                face_list.append(face)
                
                for key, detector in self.detectors.items():
                    result, image = detector.inference(frame, rect)
                    conbined_image = detector.draw_image(conbined_image)
                    result_dict[key].append(result)
                    image_dict[key].append(image)

            conbined_image_list.append(conbined_image)
        
        for key, image_list in image_dict.items():
            self.output_image(image_list, f"{video_name}/{key}")
        self.output_image(face_list, f"{video_name}/face")
        self.output_video(conbined_image_list, video_name, fps)
        return result_dict

    def output_image(self, images, output_name):
        # for test
        path = f"/app/result/images/{output_name}/"
        if not os.path.exists(path):
            os.makedirs(path)
        for idx, image in enumerate(images):
            if not image is None:
                output_path = f"{path}/{idx}.jpg"
                cv2.imwrite(output_path, image)

    def output_video(self, images, output_name, fps): 
        # for test
        path = f"/app/result/video"
        if not os.path.exists(path):
            os.makedirs(path)
        output_path = f"{path}/{output_name}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        height, width = 0, 0
        for image in images:
            if not image is None:
                height, width, _ = image.shape
                break
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print(f"Output video: {output_path}")
        for image in images:
            if not image is None:
                video_writer.write(image)

        video_writer.release()

if __name__ == "__main__":
    path = "/app/test/brightness/"
    classes = os.listdir(path)
    for cls in classes:
        result_dict_list = []
        cls_path = os.path.join(path, cls)
        videos = os.listdir(cls_path)
        for video in videos:
            video_path = os.path.join(cls_path, video)
            analyzer = Analyzer()
            result_dict = analyzer.get_video_attribute(video_path)
            result_dict_list.append(result_dict)
        pickle.dump(result_dict_list, open(f"/app/result/{cls}.pkl", "wb"))
'''