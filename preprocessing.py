### 1. 사용할 패키지 불러오기
from pycocotools.coco import COCO
import os
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import cv2

### 2. Parameter 지정
# (1) dataset_path: json 및 original image folder가 저장된 경로
# (2) original_image_folder_name: original image folder 이름 (default: JPEGImages)
# (3) mask_image_folder_name: mask image folder 이름 (default: SegmentationClass)
# (4) json_file_name: json 파일 이름
dataset_path = './dataset'
original_image_folder_name = 'JPEGImages'
mask_image_folder_name = 'SegmentationClass'
json_file_name = 'Only+Driveable+Area-1.json'

os.mkdir(os.path.join(dataset_path, mask_image_folder_name))

### 3. Mask 만들기
coco = COCO(os.path.join(dataset_path, json_file_name))
img_dir = os.path.join(dataset_path, original_image_folder_name)

for image_id in coco.imgs:
  img = coco.imgs[image_id]
  image = np.array(Image.open(os.path.join(img_dir, img['file_name'][:-4] + '.jpg')))
  cat_ids = coco.getCatIds()
  anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
  anns = coco.loadAnns(anns_ids)
  mask = coco.annToMask(anns[0])
  for i in range(len(anns)):
    mask += coco.annToMask(anns[i])
    mask[mask > 0] = 1
    cv2.imwrite(os.path.join(os.path.join(dataset_path, mask_image_folder_name), img['file_name'][:-4] + '.png'), mask)

