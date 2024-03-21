import cv2
import os
from utils.video_loader import Video_loader
from Analyzer import Analyzer

def export_excel():
    pass

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = f"{current_dir}/video"
    analyzer = Analyzer()
    videos = os.listdir(path)
    result_dict = {}
    for idx, video_name in enumerate(videos):
        result_list = []
        video_path = os.path.join(path, video_name)
        vl = Video_loader(video_path)
        while(True):
            frame = vl.get_frame()
            if frame is None:
                break
            result, merged_image = analyzer.get_brightness(frame)
            result_list.append(result)
            cv2.imwrite(f"./images/{video_name}_{idx}.jpg", merged_image)
        
        result_dict[video_name] = result_list
    
    # export_excel(result_dict, "brightness_result.xlsx")