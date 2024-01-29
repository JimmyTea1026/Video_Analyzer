from concurrent.futures import ThreadPoolExecutor
import os
from utils.video_loader import Video_loader
from attribute_detector.attribute_detector import Attribute_extractor
from bangs_detector.bangs_detector import Bangs_detector
from angle_detector.angle_detector import FaceeRotationAngleDetector
from mask_detector.mask_detector import FaceMaskDetector
from face_detector.face_detector import FaceDetector
import cv2
import dlib
import threading


def get_video_attribute(video_name):
    #---------------------------------------#
    FACE_DETECTOR = FaceDetector('dlib')
    ATTRIBUTE_EXTRACTOR = Attribute_extractor()
    ROTATION_DETECTOR = FaceeRotationAngleDetector()
    MASK_DETECTOR = FaceMaskDetector()
    BANGS_DETECTOR = Bangs_detector()
    VIDEO_LOADER = Video_loader()
    VIDEO_LOADER.load_video(f"{path}/{video_name}")
    #---------------------------------------#
    frame_count = 0
    face_list = []
    conbined_image_list = []
    detectors = {'mask': MASK_DETECTOR, 'rotate': ROTATION_DETECTOR, 
                 'bangs': BANGS_DETECTOR, 'attribute': ATTRIBUTE_EXTRACTOR}
    result_dict = {'mask': [], 'rotate': [], 'bangs': [], 'attribute': []}
    image_dict = {'mask': [], 'rotate': [], 'bangs': [], 'attribute': []}
    
    while True:
        frame = VIDEO_LOADER.get_frame()
        if frame is None:
            break
    
        frame_count += 1
        if frame_count % 3 != 0:
            continue
        
        # Face detection
        rect = FACE_DETECTOR.detect(frame)
        if rect is None: continue
        x1, y1, x2, y2 = rect
        face = frame[y1:y2, x1:x2]
        face_list.append(face)
        
        conbined_image = frame.copy()
        for key, detector in detectors.items():
            result, image = detector.inference(frame, rect)
            conbined_image = detector.draw_image(conbined_image)
            result_dict[key].append(result)
            image_dict[key].append(image)

        conbined_image_list.append(conbined_image)
    
    for key, image_list in image_dict.items():
        output_image(image_list, f"{video_name}/{key}")
        
    output_image(face_list, f"{video_name}/face")
    output_video(conbined_image_list, video_name, 5)

def get_frame_attribute(frame):
    #---------------------------------------#
    FACE_DETECTOR = FaceDetector('dlib')
    ATTRIBUTE_EXTRACTOR = Attribute_extractor()
    ROTATION_DETECTOR = FaceeRotationAngleDetector()
    MASK_DETECTOR = FaceMaskDetector()
    BANGS_DETECTOR = Bangs_detector()
    #---------------------------------------#
    result_list = []
    image_list = []
    detectors = {'mask': MASK_DETECTOR, 'rotate': ROTATION_DETECTOR, 'bangs': BANGS_DETECTOR}
    rect = FACE_DETECTOR.detect(frame)
    if rect is None: return "No face detected"
    
    conbined_image = frame.copy()
    for key, detector in detectors.items():
        result, image = detector.inference(frame, rect)
        conbined_image = detector.draw_image(conbined_image)
        result_list.append(result)
        image_list.append(image)
    
    return result_list, image_list, conbined_image


def output_image(images, output_name):
    path = f"/app/result/images/{output_name}/"
    if not os.path.exists(path):
        os.makedirs(path)
    for idx, image in enumerate(images):
        if not image is None:
            output_path = f"{path}/{idx}.jpg"
            cv2.imwrite(output_path, image)

def output_video(images, output_name, fps):
    path = f"/app/result/video/"
    if not os.path.exists(path):
        os.makedirs(path)
    output_path = f"{path}/{output_name}"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    height, width = 0, 0
    for image in images:
        if not image is None:
            height, width, _ = image.shape
            break
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for image in images:
        if not image is None:
            video_writer.write(image)

    video_writer.release()

if __name__ == "__main__":
    path = "/app/test"
    videos = os.listdir(path)
    with ThreadPoolExecutor(max_workers=5) as executor:
        future = [executor.submit(get_video_attribute, video) for video in videos]
        for future in future:
            print(future.result())