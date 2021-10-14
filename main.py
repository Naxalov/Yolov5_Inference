import torch
# from utils import draw_bbox_xyxy
import cv2
from pathlib import Path
from torch._C import dtype
from tqdm import tqdm
# Model
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
import json
def draw_coco_bbox(img,labels):
    """[coco, a bounding box is defined by four values in pixels [x_min, y_min, width, height]]

    Args:
        img (image): [numpy array]
        label (list): [x_min, y_min, width, height]
    """
    
    # w,h,_=img.shape
    for bbox in labels:

        x1,y1,w,h,c,_=bbox
        # font
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # org
        org = (int(x1), int(y1))
        
        # fontScale
        fontScale = 1
        LABELS = ['ConcreteCrack','Spalling','Efflorescene','Exposure']
        
        # Blue color in BGR
        # color_efflorescene = 

        color = (56,255,225)

        
        # Line thickness of 2 px
        thickness = 2
        
        # Using cv2.putText() method
        img = cv2.putText(img, str(c), org, font, 
                        fontScale, color, thickness, cv2.LINE_AA)


        img = cv2.rectangle(img,(int(x1),int(y1)),(int(x1+w),int(y1+h)),color,2)
    return img

model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')  # default
PATH = 'best.pt'
# Load
# model = torch.load(PATH)
# model.eval()

# Images
img_path ='103_0c0e7e72-7570-4f6c-acec-a7811fdb830d.jpg'
# imgs = ['https://ultralytics.com/images/zidane.jpg']  # batch of images
# img = cv2.imread(img_path)
# Inference
DIR_ID = '4000'
INPUT_DIR = list(Path(f'/media/buntuml/DATASET/DAMAGEAI/DATASET/103_Efflorescene/{DIR_ID}').glob('*.jpg'))
OUTPUT_DIR = Path('OUTPUT')
OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
(OUTPUT_DIR/DIR_ID).mkdir(parents=True,exist_ok=True)
for path in tqdm(INPUT_DIR):
    results = model(path)

# Results
# results.print()
    # results.save('exp')  # or .show()

# results.xyxy[0]  # img1 predictions (tensor)
    pred = results.xyxy[0].numpy().astype(float)
    annotations = {
        'pred':[]
    }
    for bbox in pred:
        inst = {
            'points':{}
        }
        inst['probability'] = bbox[4]
        inst['points']['x1']=bbox[0]
        inst['points']['y1']=bbox[1]
        inst['points']['x2']=bbox[2]
        inst['points']['y2']=bbox[3]
        annotations['pred'].append(inst)

    with open(f'{OUTPUT_DIR}/{DIR_ID}/{path.stem}.json','w',encoding='utf8') as f:        
        json.dump(annotations, f, ensure_ascii=False)




# img = draw_coco_bbox(img,pred)

# results.pandas().xyxy[0]  # img1 predictions (pandas)
# cv2.namedWindow("output", cv2.WINDOW_NORMAL)
# cv2.imshow('output',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

