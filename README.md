# Deeplab V3 Plus implementation

## 1. 가상환경 Setting
- Tensorflow 2.8
pip install tensorflow-gpu==2.8
- opencv-python
- matplotlib
- scipy
- pycocotools

## 2. Data Preprocessing
- 이미지와 COCO Json 파일로부터 Mask 생성 필요
- Mask의 각 픽셀은 1-channel로 각 class를 의미
- 0: black, 1: 1-class, 2: 2-class, ...
- 예제에서는 배경과 Lane만 있는 2-class
- preprocessing.py 파일로 mask 생성 가능
original image: jpg
mask image: png

## 3. 