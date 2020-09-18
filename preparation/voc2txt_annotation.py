import os
import argparse
import xml.etree.ElementTree as ET


def parse_arguments():
    parser = argparse.ArgumentParser(description='This a script to generate train, test, val dataset, the generated txt file will be used for yolo training')
    parser.add_argument('-name', default="VOC_2007", help="Dataset name", action="store_true")
    parser.add_argument('-input_dir',default="VOCdevkit/VOC2007", help="Read dataset annotations", action="store_true")
    parser.add_argument('-save', default='data_txt', help="Txt file generated for yolo training and test", action="store_true")
    args = parser.parse_args()
    return args

def convert_annotation(image_id, list_file):
    in_file = open(os.path.join(input_dir_path, 'Annotations/%s.xml'%(image_id)))
    tree=ET.parse(in_file)
    root = tree.getroot()
    list_file.write(os.path.join(current_path, input_dir_path, 'JPEGImages/%s.jpg'%(image_id)))
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

    list_file.write('\n')

def save_data_txt():
    for name, image_set in sets:
        print(os.path.join(input_dir_path, 'ImageSets/Main','%s.txt'%(image_set)))
        image_ids = open(os.path.join(input_dir_path, 'ImageSets/Main','%s.txt'%(image_set))).read().strip().split()
        list_file = open(os.path.join(save_path, '%s_%s.txt'%(args.name, image_set)), 'w')
        for image_id in image_ids:
            convert_annotation(image_id, list_file)
        list_file.close()

if __name__=='__main__':
    current_path = os.getcwd()
    args = parse_arguments()
    sets=[(args.name, 'train'), (args.name, 'trainval'), (args.name, 'val')]

    save_path = args.save
    input_dir_path = args.input_dir
    os.makedirs(save_path, exist_ok=True)
    classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

    save_data_txt()
