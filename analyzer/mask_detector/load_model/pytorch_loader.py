import sys
import torch
import os
# 取得當前腳本所在的目錄
current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_directory)

def load_pytorch_model(model_path):
    model = torch.load(model_path)
    return model

def pytorch_inference(model, img_arr):
    if torch.cuda.is_available():
        dev = 'cuda:0'
    else:
        dev = 'cpu'
    device = torch.device(dev)
    model.to(device)
    input_tensor = torch.tensor(img_arr).float().to(device)
    y_bboxes, y_scores, = model.forward(input_tensor)
    return y_bboxes.detach().cpu().numpy(), y_scores.detach().cpu().numpy()
