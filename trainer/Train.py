#-------------------------------------#
#       对数据集进行训练/Training
#-------------------------------------#
import os
import argparse
import numpy as np
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from preprocessing.utils_generator import Generator
from preprocessing.dataloader import yolo_dataset_collate, YoloDataset
from yolo_model.loss import YOLOLoss
from yolo_model.yolo4 import YoloBody
from tensorboardX import SummaryWriter
from tqdm import tqdm
from prettytable import PrettyTable


def parse_arguments():
    parser = argparse.ArgumentParser(description='This a script')
    parser.add_argument('-input_shape', default=(416, 416), help="Input image shape", action="store_true")
    parser.add_argument('-save', default="voc2yolo/", help="Directory stores train.txt, test.txt",
                        action="store_true")
    parser.add_argument('-cosine_lr', default=True, help="Activate the cosine learning rate",
                        action="store_true")
    parser.add_argument('-mosaic_augmentation', default=True, help="Activate the data augmentation method: mosaic",
                        action="store_true")
    parser.add_argument('-cuda', default=True, help="Activate the GPUs for training",
                        action="store_true")
    parser.add_argument('-smooth_label', default=0, help="Set the coefficient for smooth label",
                        action="store_true")
    parser.add_argument('-data_loader', default=True, help="Whether to use Data Loader",
                        action="store_true")
    parser.add_argument('-annotation_path', default='../preparation/data_txt/VOC_2007_trainval.txt',
                        help="Trainval.txt annotation path", action="store_true")
    parser.add_argument('-anchors_path', default='../preparation/data_txt/yolo_anchors_kmeans.txt',
                        help="Path to the file yolo_anchors.txt",
                        action="store_true")
    parser.add_argument('-classes_path', default='../preparation/data_txt/voc_classes.txt',
                        help="Path to the file voc_classes.txt",
                        action="store_true")
    parser.add_argument('-val_percent', default=0.05, help="Percentage of validation data in the total data",
                        action="store_true")
    parser.add_argument('-model_path', default="../models/original/yolo4_voc_weights.pth",
                        help="Load the pretrained model",
                        action="store_true")
    args = parser.parse_args()
    return args

#---------------------------------------------------#
#   获得类和先验框/Get the data classes and anchors
#---------------------------------------------------#
def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape([-1,3,2])[::-1,:,:]

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit_ont_epoch(net,yolo_losses,epoch,epoch_size,epoch_size_val,gen,genval,Epoch,cuda,writer):
    total_loss = 0
    val_loss = 0
    start_time = time.time()
    with tqdm(total=epoch_size,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_size:
                break
            images, targets = batch[0], batch[1]
            with torch.no_grad():
                if cuda:
                    images = Variable(torch.from_numpy(images).type(torch.FloatTensor)).cuda()
                    targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]
                else:
                    images = Variable(torch.from_numpy(images).type(torch.FloatTensor))
                    targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]
            optimizer.zero_grad()
            outputs = net(images)
            losses = []
            for i in range(3):
                loss_item = yolo_losses[i](outputs[i], targets)
                losses.append(loss_item[0])
            loss = sum(losses)
            loss.backward()
            optimizer.step()
            # 将loss写入tensorboard，每一步都写
            writer.add_scalar('Train_loss', loss, (epoch*epoch_size + iteration))

            total_loss += loss
            waste_time = time.time() - start_time
            
            pbar.set_postfix(**{'total_loss': total_loss.item() / (iteration + 1), 
                                'lr'        : get_lr(optimizer),
                                'step/s'    : waste_time})
            pbar.update(1)


            start_time = time.time()
        
    # 将loss写入tensorboard，下面注释的是每个世代保存一次
    # writer.add_scalar('Train_loss', total_loss/(iteration+1), epoch)

    print('Start Validation')
    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(genval):
            if iteration >= epoch_size_val:
                break
            images_val, targets_val = batch[0], batch[1]

            with torch.no_grad():
                if cuda:
                    images_val = Variable(torch.from_numpy(images_val).type(torch.FloatTensor)).cuda()
                    targets_val = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets_val]
                else:
                    images_val = Variable(torch.from_numpy(images_val).type(torch.FloatTensor))
                    targets_val = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets_val]
                optimizer.zero_grad()
                outputs = net(images_val)
                losses = []

                for i in range(3):
                    loss_item = yolo_losses[i](outputs[i], targets_val)
                    losses.append(loss_item[0])
                loss = sum(losses)
                val_loss += loss
                # 将loss写入tensorboard, 下面注释的是每一步都写
                # writer.add_scalar('Val_loss',val_loss/(epoch_size_val+1), (epoch*epoch_size_val + iteration))

            pbar.set_postfix(**{'total_loss': val_loss.item() / (iteration + 1)})
            pbar.update(1)
            
    # 将loss写入tensorboard，每个世代保存一次
    writer.add_scalar('Val_loss',val_loss/(epoch_size_val+1), epoch)
    print('Finish Validation')
    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss/(epoch_size+1),val_loss/(epoch_size_val+1)))

    print('Saving state, iter:', str(epoch+1))
    torch.save(model.state_dict(), 'logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth'%((epoch+1),total_loss/(epoch_size+1),val_loss/(epoch_size_val+1)))


if __name__ == "__main__":
    args = parse_arguments()
    table = PrettyTable()
    table.field_names = ['Parts', 'Names', 'Input contents', 'Check?']
    #-------------------------------#
    #   输入的shape大小
    #   显存比较小可以使用416x416
    #   显存比较大可以使用608x608
    #-------------------------------#
    input_shape = args.input_shape
    # 6.4f 表示输出的为六位整数位和4位小数位
    table.add_row([' ', 'input shape', str(input_shape), 'check'])

    #-------------------------------#
    #   tricks的使用设置
    #-------------------------------#
    Cosine_lr = args.cosine_lr
    mosaic = args.mosaic_augmentation
    # 用于设定是否使用cuda
    Cuda = args.cuda
    smooth_label = args.smooth_label
    table.add_row(['Trick settings', 'cosine lr', Cosine_lr, 'check'])
    table.add_row(['Trick settings', 'mosaic', mosaic, 'check'])
    table.add_row(['Trick settings', 'cuda avaliable', torch.cuda.is_available(), 'check'])
    table.add_row(['Trick settings', 'use cuda', Cuda, 'check'])
    table.add_row(['Trick settings', 'use smooth label', smooth_label, 'check'])

    #-------------------------------#
    #   Dataloder的使用
    #-------------------------------#
    Use_Data_Loader = args.data_loader
    annotation_path = args.annotation_path
    table.add_row(['Data settings', 'use data loader', Use_Data_Loader, 'check'])
    table.add_row(['Data settings', 'data path', annotation_path, 'check'])

    #-------------------------------#
    #   获得先验框和类
    #-------------------------------#
    anchors_path = args.anchors_path
    classes_path = args.classes_path
    class_names = get_classes(classes_path)
    anchors = get_anchors(anchors_path)
    num_classes = len(class_names)

    table.add_row(['Data settings', 'anchors path', anchors_path, 'check'])
    #table.add_row(['Data settings', 'anchor values', anchors, 'check'])
    table.add_row(['Data settings', 'anchor numbers', str(len(anchors[0])), 'check'])

    table.add_row(['Data settings', 'class path', classes_path, 'check'])
    #table.add_row(['Data settings', 'class names' , class_names, 'check'])
    table.add_row(['Data settings', 'class numbers', num_classes, 'check'])

    table.add_row(['Data settings', 'Final layer output channels', len(anchors[0])*(5 + num_classes), 'check'])

    # -------------------------------#
    #    # 创建模型
    # -------------------------------#
    model = YoloBody(len(anchors[0]),num_classes)
    model_path = args.model_path
    table.add_row(['Model settings', 'model path', model_path, 'check'])


    # 加快模型训练的效率
    # print('Loading weights into state dict...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path, map_location=device)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) ==  np.shape(v)}
    """
        yolo_head3.1.weight model torch.Size([75, 256, 1, 1]) trained torch.Size([255, 256, 1, 1])
        yolo_head3.1.bias model torch.Size([75]) trained torch.Size([255])
        yolo_head2.1.weight model torch.Size([75, 512, 1, 1]) trained torch.Size([255, 512, 1, 1])
        yolo_head2.1.bias model torch.Size([75]) trained torch.Size([255])
        yolo_head1.1.weight model torch.Size([75, 1024, 1, 1]) trained torch.Size([255, 1024, 1, 1])
        yolo_head1.1.bias model torch.Size([75]) trained torch.Size([255])
    """
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    # print('Finished!')

    net = model.train()

    if Cuda:
        net = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        net = net.cuda()

    # 建立loss函数
    yolo_losses = []
    for i in range(3):
        yolo_losses.append(YOLOLoss(np.reshape(anchors,[-1,2]),num_classes, \
                                (input_shape[1], input_shape[0]), smooth_label, Cuda))

    # 0.1用于验证，0.9用于训练
    val_split = args.val_percent
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val
    table.add_row(['Train settings', 'train dataset', num_train, 'check'])
    table.add_row(['Train settings', 'val dataset', num_val, 'check'])

    writer = SummaryWriter(log_dir='logs',flush_secs=60)
    if Cuda:
        graph_inputs = torch.from_numpy(np.random.rand(1,3,input_shape[0],input_shape[1])).type(torch.FloatTensor).cuda()
    else:
        graph_inputs = torch.from_numpy(np.random.rand(1,3,input_shape[0],input_shape[1])).type(torch.FloatTensor)
    writer.add_graph(model, (graph_inputs,))

    if True:
        lr = 1e-3
        Batch_size = 4
        Init_Epoch = 0
        Freeze_Epoch = 50
        table.add_row(['Parts frozen', 'initial lr', lr, 'check'])
        table.add_row(['Parts frozen', 'batch size', Batch_size, 'check'])
        table.add_row(['Parts frozen', 'initial epoch', Init_Epoch, 'check'])
        table.add_row(['Parts frozen', 'freeze epoch', Freeze_Epoch, 'check'])
        table.add_row(['Parts frozen', 'optimizer', 'Adam', 'check'])
        print(table)

        optimizer = optim.Adam(net.parameters(),lr,weight_decay=5e-4)

        if Cosine_lr:
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
        else:
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.95)

        if Use_Data_Loader:
            train_dataset = YoloDataset(lines[:num_train], (input_shape[0], input_shape[1]), mosaic=mosaic)
            val_dataset = YoloDataset(lines[num_train:], (input_shape[0], input_shape[1]), mosaic=False)
            gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=4, pin_memory=True,
                                    drop_last=True, collate_fn=yolo_dataset_collate)
            gen_val = DataLoader(val_dataset, batch_size=Batch_size, num_workers=4,pin_memory=True, 
                                    drop_last=True, collate_fn=yolo_dataset_collate)
        else:
            gen = Generator(Batch_size, lines[:num_train],
                            (input_shape[0], input_shape[1])).generate(mosaic = mosaic)
            gen_val = Generator(Batch_size, lines[num_train:],
                            (input_shape[0], input_shape[1])).generate(mosaic = False)

        epoch_size = max(1, num_train//Batch_size)
        epoch_size_val = num_val//Batch_size
        #------------------------------------#
        #   冻结一定部分训练
        #------------------------------------#
        for param in model.backbone.parameters():
            param.requires_grad = False

        for epoch in range(Init_Epoch,Freeze_Epoch):
            fit_ont_epoch(net,yolo_losses,epoch,epoch_size,epoch_size_val,gen,gen_val,Freeze_Epoch,Cuda,writer)
            lr_scheduler.step()

    if True:
        lr = 1e-4
        Batch_size = 2
        Freeze_Epoch = 50
        Unfreeze_Epoch = 100
        table.add_row(['Released', 'initial lr', lr, 'check'])
        table.add_row(['Released', 'batch size', Batch_size, 'check'])
        table.add_row(['Released', 'freeze epoch', Freeze_Epoch, 'check'])
        table.add_row(['Released', 'unfreeze epoch', Unfreeze_Epoch, 'check'])
        table.add_row(['Released', 'optimizer', 'Adam', 'check'])
        print(table)

        optimizer = optim.Adam(net.parameters(),lr,weight_decay=5e-4)
        if Cosine_lr:
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
        else:
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.95)

        if Use_Data_Loader:
            train_dataset = YoloDataset(lines[:num_train], (input_shape[0], input_shape[1]), mosaic=mosaic)
            val_dataset = YoloDataset(lines[num_train:], (input_shape[0], input_shape[1]), mosaic=False)
            gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=4, pin_memory=True,
                                    drop_last=True, collate_fn=yolo_dataset_collate)
            gen_val = DataLoader(val_dataset, batch_size=Batch_size, num_workers=4,pin_memory=True, 
                                    drop_last=True, collate_fn=yolo_dataset_collate)
        else:
            gen = Generator(Batch_size, lines[:num_train],
                            (input_shape[0], input_shape[1])).generate(mosaic = mosaic)
            gen_val = Generator(Batch_size, lines[num_train:],
                            (input_shape[0], input_shape[1])).generate(mosaic = False)

        epoch_size = max(1, num_train//Batch_size)
        epoch_size_val = num_val//Batch_size
        #------------------------------------#
        #   解冻后训练
        #------------------------------------#
        for param in model.backbone.parameters():
            param.requires_grad = True

        for epoch in range(Freeze_Epoch,Unfreeze_Epoch):
            fit_ont_epoch(net,yolo_losses,epoch,epoch_size,epoch_size_val,gen,gen_val,Unfreeze_Epoch,Cuda,writer)
            lr_scheduler.step()
