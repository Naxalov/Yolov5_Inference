#pip install -qr https://raw.githubusercontent.com/ultralytics/yolov5/master/requirements.txt  # install dependencies
import torch

from main import OUTPUT_DIR

PATH = 'best.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=PATH)  # default

img_path ='103_0c7c31ff-91c2-4944-9e8c-b02a8c43b667.jpg'

results = model(img_path)
OUTPUT_DIR = '1'
results.save(OUTPUT_DIR)
# print(results.xyxy[0]) 