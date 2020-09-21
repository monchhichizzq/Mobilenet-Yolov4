#-------------------------------------#
#       mAP所需文件计算代码 Get detection results
#-------------------------------------#
import cv2
import numpy as np
import colorsys
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from yolo_model.yolo_test import YOLO
from yolo_model.yolo4 import YoloBody
from PIL import Image,ImageFont, ImageDraw
from yolo_model.utils import non_max_suppression, bbox_iou, DecodeBox, letterbox_image,yolo_correct_boxes
from tqdm import tqdm

class mAP_Yolo(YOLO):
    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self,image_id,image):
        self.confidence = 0.01
        f = open("./input/detection-results/"+image_id+".txt","w") 
        image_shape = np.array(np.shape(image)[0:2])

        crop_img = np.array(letterbox_image(image, (self.model_image_size[0],self.model_image_size[1]))) # Resize the image
        print(np.shape(image), np.shape(crop_img), self.model_image_size[0], self.model_image_size[1])
        cv2.imshow('s', crop_img)
        cv2.waitKey(1)


        photo = np.array(crop_img,dtype = np.float32)
        photo /= 255.0
        photo = np.transpose(photo, (2, 0, 1))
        photo = photo.astype(np.float32)
        images = []
        images.append(photo)
        images = np.asarray(images)

        with torch.no_grad():
            images = torch.from_numpy(images)
            if self.cuda:
                images = images.cuda()
            outputs = self.net(images)
            print('outputs 13', outputs[0].shape)
            print('outputs 26', outputs[1].shape)
            print('outputs 52', outputs[2].shape)
            
        output_list = []
        for i in range(3):
            output_list.append(self.yolo_decodes[i](outputs[i])) #
            #  Happened in yolo.py:
            # self.yolo_decodes.append(DecodeBox(self.anchors[i], len(self.class_names),  (self.model_image_size[1], self.model_image_size[0])))
            # DecodeBox could give us the new outputs adjusted for each scale outputs and it would be in shape  [1, 3*13*13, 85]

        output = torch.cat(output_list, 1) # [1, 10647, 85]
        print('output', np.shape(output))
        batch_detections = non_max_suppression(output, len(self.class_names),
                                                conf_thres=self.confidence,
                                                nms_thres=self.iou)
                            # (x1, y1, x2, y2, obj_conf, class_conf, class_pred)


        try:
            batch_detections = batch_detections[0].cpu().numpy()
            print('batch_detections', np.shape(batch_detections)) # (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        except:
            return image
            
        top_index = batch_detections[:,4]*batch_detections[:,5] > self.confidence # Find all the index where obj_conf*class_conf > confidence
        top_conf = batch_detections[top_index,4]*batch_detections[top_index,5]   # Output the index
        top_label = np.array(batch_detections[top_index,-1],np.int32) # Find the class_pred through index
        top_bboxes = np.array(batch_detections[top_index,:4])   # Find the bounding box through index
        top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:,0],-1),np.expand_dims(top_bboxes[:,1],-1),np.expand_dims(top_bboxes[:,2],-1),np.expand_dims(top_bboxes[:,3],-1)

        # 去掉灰条
        boxes = yolo_correct_boxes(top_ymin,top_xmin,top_ymax,top_xmax,np.array([self.model_image_size[0],self.model_image_size[1]]),image_shape)
        # (x1, y1, x2, y2)

        for i, c in enumerate(top_label):
            predicted_class = self.class_names[c] # Find the class name through label_index
            score = str(top_conf[i])  # obj_conf*class_conf
            top, left, bottom, right = boxes[i]
            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

        f.close()
        return 

yolo = mAP_Yolo()
VOC_2007_test_path = '../preparation/VOCdevkit/VOCtest_06-Nov-2007/VOCdevkit/VOC2007'
image_ids = open(os.path.join(VOC_2007_test_path,'ImageSets/Main/test.txt')).read().strip().split()

if not os.path.exists("./input"):
    os.makedirs("./input")
if not os.path.exists("./input/detection-results"):
    os.makedirs("./input/detection-results")
if not os.path.exists("./input/images-optional"):
    os.makedirs("./input/images-optional")


for image_id in tqdm(image_ids):
    print(image_id)
    image_path = os.path.join(VOC_2007_test_path, 'JPEGImages', image_id+".jpg")
    image = Image.open(image_path)
    # 开启后在之后计算mAP可以可视化
    image.save("./input/images-optional/"+image_id+".jpg")
    yolo.detect_image(image_id,image)

print("Conversion completed!")

