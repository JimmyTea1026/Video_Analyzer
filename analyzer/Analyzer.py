from brightness_detector.brightness_detector import Brightness_detector
from bangs_detector.bangs_detector import Bangs_detector
from angle_detector.angle_detector import FaceeRotationAngleDetector
from mask_detector.mask_detector import FaceMaskDetector
from face_detector.face_detector import FaceDetector

class Analyzer:
    def __init__(self) -> None:
        BRIGHTNESS_DETECTOR = Brightness_detector()
        ROTATION_DETECTOR = FaceeRotationAngleDetector()
        MASK_DETECTOR = FaceMaskDetector()
        BANGS_DETECTOR = Bangs_detector()
        self.detectors = {'mask': MASK_DETECTOR, 'rotate': ROTATION_DETECTOR, 
                          'bangs': BANGS_DETECTOR, 'brightness': BRIGHTNESS_DETECTOR}
        self.FACE_DETECTOR = FaceDetector('scrfd')
    
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
            # raise ValueError("No face detected")
            rect = self.FACE_DETECTOR.detect(frame)
            if rect is None:
                return "No face detected"
    
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
    
    def get_brightness(self, frame):
        rect = self.FACE_DETECTOR.detect(frame)
        if rect is None:
            raise ValueError("No face detected")
                
        masked_frame, merged_frame = self.detectors['bangs'].get_masked_frame(frame, rect)
        result = self.detectors['brightness'].get_range(masked_frame)
        
        return result, merged_frame