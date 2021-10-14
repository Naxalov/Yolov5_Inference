from pathlib import Path
from PIL import Image
import cv2
from tqdm import tqdm
import json
import shutil

def draw_coco_bbox(img,labels):
    """[coco, a bounding box is defined by four values in pixels [x_min, y_min, width, height]]

    Args:
        img (image): [numpy array]
        label (list): [x_min, y_min, width, height]
    """
    
    # w,h,_=img.shape
    for bbox in labels:

        x1 = int(bbox['points']['x1'])
        y1 = int(bbox['points']['y1'])
        x2 = int(bbox['points']['x2'])
        y2 = int(bbox['points']['y2'])
        c = str(bbox['probability'])
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


        img = cv2.rectangle(img,(int(x1),int(y1)),(int(x2),int(y2)),color,2)
    return img

def read_label(path):
    try:
        f = open(path).read().encode('utf8')
        data = json.loads(f)
    except:
        return False
    return data


DIR_ID = '4000'

IMG_DIR = Path(f'/media/buntuml/DATASET/DAMAGEAI/DATASET/103_Efflorescene/{DIR_ID}')

input_labels = list(Path(f'OUTPUT/{DIR_ID}').glob('*.json'))

data = read_label(input_labels[0])

img_path = (IMG_DIR/input_labels[0].stem).with_suffix('.jpg')
img = cv2.imread(str(img_path))
img= draw_coco_bbox(img,data['pred'])
cv2.imwrite('1.jpg',img)

# points = data['pred'][0]['points']