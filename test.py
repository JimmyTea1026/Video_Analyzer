from analyzer.analyzer import Analyzer
from analyzer.face_detector.face_detector import FaceDetector
from StabilizationAlgorithm import StabilizationAlgorithm
from openpyxl import Workbook
import os
import cv2

def mask_detect(analyzer, frame, rect):
    x1, y1, x2, y2 = rect
    h = y2 - y1
    w = x2 - x1
    if h < 50 or w < 50:
        return False
    model_result = analyzer.inference_mask(frame)
    return model_result

def face_detect(frame):
    face_detector = FaceDetector('scrfd')
    rect = face_detector.detect(frame)
    return rect

def image_combination(frame, rect, model_result, algo_result):
    model_image = draw_image(frame, rect, model_result)
    algo_image = draw_image(frame, rect, algo_result)
    combined_image = cv2.vconcat([model_image, algo_image])
    cv2.putText(combined_image, "Mask Detection", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    # cv2.imwrite("result.jpg", combined_image)
    return combined_image

def draw_image(frame, rect, result):
    draw_frame = frame.copy()
    color = (0, 255, 0) if result else (0, 0, 255)
    x1, y1, x2, y2 = rect
    cv2.rectangle(draw_frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(draw_frame, "Mask" if result else "No Mask", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return draw_frame

def print_result(frame_count, model_result, algo_result):
    print(f"Frame {frame_count}: Mask: {model_result}, Stabilization: {algo_result}")

def video_generate(video_name, combined_image_list):
    path = "./mask_test"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    width, height = combined_image_list[0].shape[1], combined_image_list[0].shape[0]
    fps = 5.0
    out = cv2.VideoWriter(f'{path}/{video_name}_result.mp4', fourcc, fps, (width, height))
    for image in combined_image_list:
        out.write(image)
    out.release()

def queue_size_test(model_result_list, ws, idx):
    result_size = len(model_result_list)
    ws.cell(row=1, column=1, value="Result Size")
    ws.cell(row=1, column=2, value="Model Accuracy")
    titles = ["Queue Size {}".format(i) for i in range(3, 11)]
    for i, title in enumerate(titles, start=3):  # 從第三列開始
        ws.cell(row=1, column=i, value=title)
        
    model_acc = sum(1 for _ in iter(model_result_list) if _ == True) / result_size
    model_acc = round(model_acc, 2)
    print(f"Model accuracy: {model_acc}")
    ws.cell(row=idx+2, column=1, value=result_size)
    ws.cell(row=idx+2, column=2, value=model_acc)
    for i in range(3, 11):
        algorithm = StabilizationAlgorithm(q_size=i)
        algo_result_list = []
        for result in model_result_list:
            algo_result_list.append(algorithm.get_algorithm_result(result))
        algo_acc = sum(1 for _ in iter(algo_result_list) if _ == True) / result_size
        algo_acc = round(algo_acc, 2)
        print(f"Queue size: {i} Algorithm accuracy: {algo_acc}")
        ws.cell(row=idx+2, column=i, value=algo_acc)
            

def video_process(video_path):
    analyzer = Analyzer()
    algorithm = StabilizationAlgorithm(q_size=5)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    combined_image_list = []
    model_result_list = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % 3 != 0: continue  # 每3幀處理一次
        
        rect = face_detect(frame)
        if rect is None:
            print(f"Frame {frame_count}: No face detected")
            continue
        model_result = mask_detect(analyzer, frame, rect)
        algo_result = algorithm.get_algorithm_result(model_result)
        
        # print_result(frame_count, model_result, algo_result)
        combined_image = image_combination(frame, rect, model_result, algo_result)
        combined_image_list.append(combined_image)
        model_result_list.append(model_result)
        
    # video_name = video_path.split('/')[-1].split('.')[0]
    # video_generate(video_name, combined_image_list)
    cap.release()
    return model_result_list

if __name__ == "__main__":
    wb = Workbook()
    ws = wb.active
    video_dir_path = "test/Mask/mask"
    videos = os.listdir(video_dir_path)
    for idx, video in enumerate(videos):
        video_path = os.path.join(video_dir_path, video)
        print(f"Processing {video_path}")
        model_result_list = video_process(video_path)
        queue_size_test(model_result_list, ws, idx)
        wb.save("mask_detection_result.xlsx")