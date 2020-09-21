#-------------------------------------#
# Use voc2yolo4 to split the train, test data
# Get the ground truth of test data
#-------------------------------------#
import sys
import os
import glob
import xml.etree.ElementTree as ET
from tqdm import tqdm

VOC_2007_test_path = '../preparation/VOCdevkit/VOCtest_06-Nov-2007/VOCdevkit/VOC2007'
# image_ids = open('VOCdevkit/VOC2007/ImageSets/Main/test.txt').read().strip().split()
image_ids = open(os.path.join(VOC_2007_test_path,'ImageSets/Main/test.txt')).read().strip().split()

if not os.path.exists("./input"):
    os.makedirs("./input")
if not os.path.exists("./input/ground-truth"):
    os.makedirs("./input/ground-truth")

for image_id in tqdm(image_ids):
    with open("./input/ground-truth/"+image_id+".txt", "w") as new_f:
        root = ET.parse(os.path.join(VOC_2007_test_path, 'Annotations', image_id+".xml")).getroot()
        for obj in root.findall('object'):
            if obj.find('difficult')!=None:
                difficult = obj.find('difficult').text
                if int(difficult)==1:
                    continue
            obj_name = obj.find('name').text
            bndbox = obj.find('bndbox')
            left = bndbox.find('xmin').text
            top = bndbox.find('ymin').text
            right = bndbox.find('xmax').text
            bottom = bndbox.find('ymax').text
            new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
print("Conversion completed!")
