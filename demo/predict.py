#-------------------------------------#
#       对单张图片进行预测
#-------------------------------------#
from yolo_model.yolo_test import YOLO
from PIL import Image
import cv2
import numpy as np


yolo = YOLO()

# while True:
#     img = input('Input image filename:')
#     try:
#         image = Image.open(img)
#         # print(image)
#     except:
#         print('Open Error! Try again!')
#         continue
#     else:
#         r_image = yolo.detect_image(image)
#         r_image.show()

cap = cv2.VideoCapture("test_files/test_wall.mp4")
while True:
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #print(np.max(frame))
    #cv2.imshow('', frame)
    #cv2.waitKey(1)

    image = Image.fromarray(frame.astype('uint8'), mode='RGB')
    r_image = yolo.detect_image(image)
    # r_image.show()
    SHOW_IMG = np.array(r_image)
    SHOW_IMG= cv2.cvtColor(SHOW_IMG, cv2.COLOR_RGB2BGR)
    cv2.imshow('video', SHOW_IMG)
    cv2.waitKey(1)


