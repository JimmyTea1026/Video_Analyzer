import os
from utils.video_loader import Video_loader
from utils.attribute_extractor import Attribute_extractor
from utils.bangs_detector import Bangs_detector
from FaceAngleRotation.rotation_cal import FaceeRotationAngleDetector
from mask_detector.mask_detector import FaceMaskDetector
from utils.scrfd import SCRFD
import cv2
import dlib
import threading

EXTRACTOR = Attribute_extractor()
ROTATION_DETECTOR = FaceeRotationAngleDetector()
MASK_DETECTOR = FaceMaskDetector()
BANGS_DETECTOR = Bangs_detector()
VIDEO_LOADER = Video_loader()
FACE_DETECTOR = SCRFD("/utlis/model/det_500m.onnx")
FACE_DETECTOR.prepare(0)

def get_video_attribute(video_name):
    EXTRACTOR.reset()
    ROTATION_DETECTOR.reset()
    MASK_DETECTOR.reset()
    BANGS_DETECTOR.reset()
    face_list = []
    frame_count = 0
    conbined_image_list = []
    while True:
        frame = VIDEO_LOADER.get_frame()
        if frame is None:
            break
        conbined_image = frame.copy()
        frame_count += 1
        if frame_count % 3 != 0:
            continue
        
        threads = []
        # Get mask detection
        mask_detect_thread = threading.Thread(target=MASK_DETECTOR.inference, args=(frame,))
        threads.append(mask_detect_thread)
        mask_detect_thread.start()

        # Face detection
        face, rect = face_detection(frame)
        face_list.append(face)
        
        if not face is None:
            # Get face rotate angle
            rotation_thread = threading.Thread(target=ROTATION_DETECTOR.inference, args=(frame, rect))
            threads.append(rotation_thread) 
            rotation_thread.start()
        
            # Get brightness
            brightness_thread = threading.Thread(target=EXTRACTOR.inference, args=(face,))
            threads.append(brightness_thread)
            brightness_thread.start()
            
            # Get bangs
            bangs_thread = threading.Thread(target=BANGS_DETECTOR.inference, args=(face,))
            threads.append(bangs_thread)
            bangs_thread.start()

        for t in threads:
            t.join()
        
        conbined_image = MASK_DETECTOR.draw_image(conbined_image, False)
        conbined_image = ROTATION_DETECTOR.draw_image(conbined_image, False)
        conbined_image = BANGS_DETECTOR.merge_image(conbined_image, overlap=True)
        conbined_image_list.append(conbined_image)
        MASK_DETECTOR.clear_data()
        ROTATION_DETECTOR.clear_data()
        BANGS_DETECTOR.clear_data()
        
    bangs_image_list = BANGS_DETECTOR.get_image_list()
    mask_image_list = MASK_DETECTOR.get_image_list()
    rotate_image_list = ROTATION_DETECTOR.image_list
    output_image(mask_image_list, f"{video_name}/mask")
    output_image(rotate_image_list, f"{video_name}/rotate")
    output_image(bangs_image_list, f"{video_name}/bangs")
    output_image(face_list, f"{video_name}/face")
    
    output_video(conbined_image_list, video_name, 10)
    # output_video(rotate_images, "rotate", 30)
    # output_video(face_list, "face", 30)

def face_detection(frame, mode='scrfd'):
    if mode == 'haar':
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_classifier = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        
        bbox = face_classifier.detectMultiScale(
            gray_frame, scaleFactor=1.1, minNeighbors=6, minSize=(80, 80)
        )
        face = None
        if len(bbox) != 0:
            x, y, w, h = bbox[0][0], bbox[0][1], bbox[0][2], bbox[0][3]
            face = frame[y:y+h, x:x+w]
            # cv2.imwrite('test.jpg', face)
        return face
    elif mode == 'dlib':
        face_detector = dlib.get_frontal_face_detector()
        rects = face_detector(frame, 1)
        if len(rects) == 0:
            return None, None
        else:
            rect = rects[0]
            x1 = rect.left()
            y1 = rect.top()
            x2 = rect.right()
            y2 = rect.bottom()
            face = frame[y1:y2, x1:x2]
            return face, rect
    else:
        bboxes, kpss = FACE_DETECTOR.autodetect(frame, max_num=1)


def output_image(images, output_name):
    path = f"/app/images/{output_name}/"
    if not os.path.exists(path):
        os.makedirs(path)
    for idx, image in enumerate(images):
        if not image is None:
            output_path = f"{path}/{idx}.jpg"
            cv2.imwrite(output_path, image)

def output_video(images, output_name, fps):
    output_path = f"/app/video/{output_name}"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 10
    height, width, _ = images[0].shape
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for image in images:
        if not image is None:
            video_writer.write(image)

    video_writer.release()

if __name__ == "__main__":
    path = "/app/test"
    videos = os.listdir(path)
    for i, video in enumerate(videos):
        VIDEO_LOADER.load_video(f"{path}/{video}")
        get_video_attribute(video)
        if i == 5:
            break

    # video_loader.load_video("/app/test.mp4")
    # get_video_attribute()