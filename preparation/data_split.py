import os
import random 
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='This a script')
    parser.add_argument('-xml', default="VOCdevkit/VOC2007/Annotations", help="Read dataset annotations", action="store_true")
    parser.add_argument('-save',default="voc2yolo/", help="Directory stores train.txt, test.txt", action="store_true")
    parser.add_argument('-test_percent', default=0.05, help="Percentage of test data in the total data", action="store_true")
    parser.add_argument('-val_percent', default=0.05, help="Percentage of validation data in the total data",
                        action="store_true")
    args = parser.parse_args()
    return args

def transition():
    # Get all the xml files
    total_xml = []
    for xml in os.listdir(xmlfilepath):
        if xml.endswith(".xml"):
            total_xml.append(xml)

    xml_num = len(total_xml)
    list = range(xml_num)
    trainval_num = int(xml_num * trainval_percent)
    train_num = int(trainval_num * train_percent)
    trainval = random.sample(list, trainval_num)
    train = random.sample(trainval, train_num)

    print("Total size: {0}, Training: {1}, Validation: {2}, Test: {3}".format(xml_num,
                                                                              train_num,
                                                                              (trainval_num-train_num),
                                                                              (xml_num-trainval_num)))

    ftrainval = open(os.path.join(saveBasePath, 'trainval.txt'), 'w')
    ftest = open(os.path.join(saveBasePath, 'test.txt'), 'w')
    ftrain = open(os.path.join(saveBasePath, 'train.txt'), 'w')
    fval = open(os.path.join(saveBasePath, 'val.txt'), 'w')

    for i in list:
        name = total_xml[i][:-4] + '\n' # 000005.xml -> 000005
        if i in trainval:
            ftrainval.write(name)
            if i in train:
                ftrain.write(name)
            else:
                fval.write(name)
        else:
            ftest.write(name)

    ftrainval.close()
    ftrain.close()
    fval.close()
    ftest.close()


if __name__== '__main__':
    args = parse_arguments()
    xmlfilepath = args.xml
    saveBasePath = args.save
    os.makedirs(saveBasePath, exist_ok=True)

    trainval_percent = 1 - args.test_percent
    train_percent = 1 - args.val_percent
    transition()



