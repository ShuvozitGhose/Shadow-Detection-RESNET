import random
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision.transforms.functional as F
import argparse
import pickle
import os
import time
from random import randint
import torchvision
from torch import nn
import math
from torch.utils.checkpoint import checkpoint
import torch.optim as optim
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from matplotlib import pyplot as plt



class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out




def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)



class TransBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, upsample=None, **kwargs):
        super(TransBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        if upsample is not None and stride != 1:
            self.conv2 = nn.ConvTranspose2d(inplanes, planes,
                                            kernel_size=3, stride=stride, padding=1,
                                            output_padding=1, bias=False)
        else:
            self.conv2 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            residual = self.upsample(x)

        out += residual
        out = self.relu(out)

        return out



class ShadowNet(nn.Module):
    def __init__(self, pretrained=False):


        super(ShadowNet, self).__init__()
        block = Bottleneck  #1x1, 3x3, 1x1 enocder block
        transblock = TransBasicBlock # 2 convolution + 1 deconvolutoin decoder block
        layers = [3, 4, 6, 3] #number of blocks per layer
        
        
        # Encoder part
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.conv5 = nn.Conv2d(2048, 512, kernel_size=1, bias=False)
        
        
        #Decoder part
        self.inplanes = 512
        self.deconv1 = self._make_transpose(transblock, 256, 6, stride=2)
        self.deconv2 = self._make_transpose(transblock, 128, 4, stride=2)
        self.deconv3 = self._make_transpose(transblock, 64, 3, stride=2)
        self.deconv4 = self._make_transpose(transblock, 64, 3, stride=2)
        
        # final block
        self.inplanes = 64
        self.final_conv = self._make_transpose(transblock, 64, 3)

        self.upsam= nn.ConvTranspose2d(64, 32,
                                   kernel_size=2, stride=2,
                                   padding=0, bias=False)

        self.conv6 = conv3x3(32, 16)
        self.conv7 = conv3x3(16, 1)


        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []

        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    
    def _make_transpose(self, block, planes, blocks, stride=1):

        upsample = None
        if stride != 1:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.inplanes, planes,
                                   kernel_size=2, stride=stride,
                                   padding=0, bias=False),
                nn.BatchNorm2d(planes),
            )
        elif self.inplanes != planes:
            upsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []

        for i in range(1, blocks):
            layers.append(block(self.inplanes, self.inplanes))

        layers.append(block(self.inplanes, planes, stride, upsample))
        self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, input):
        # Encoder forward
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        #print(x.size())
        x = self.layer1(x)
        #print(x.size())
        x = self.layer2(x)
        #print(x.size())
        x = self.layer3(x)
        #print(x.size())
        x = self.layer4(x)
        #print(x.size())
        x = self.conv5(x)
        #print(x.size())

        # Decoder forward
        x = self.deconv1(x)
        #print(x.size())
        x = self.deconv2(x)
        #print(x.size())
        x = self.deconv3(x)
        #print(x.size())
        x = self.deconv4(x)
        #print(x.size())

        # final convolution
        x = self.final_conv(x)
        #print(x.size())
        x = self.upsam(x)
        #print(x.size())
        x = self.conv6(x)
        #print(x.size())
        x = self.conv7(x)
        #print(x.size())

        return x


def get_ransform(opt):
    transform_list = []
    if opt.Train:
        transform_list.extend([transforms.ToTensor()])
    else:
        transform_list.extend([transforms.ToTensor()])
    # transform_list.extend(
    #    [transforms.ToTensor()])
    return transforms.Compose(transform_list)


class CreateDataset_shadow(data.Dataset):
    def __init__(self, opt):

        self.opt = opt
        self.Train_img_List = sorted(os.listdir(opt.Train_img_dir))
        self.Train_label_List = sorted(os.listdir(opt.Train_label_dir))
        self.Test_img_List = sorted(os.listdir(opt.Test_img_dir))
        self.Test_label_List = sorted(os.listdir(opt.Test_label_dir))

        self.transform = get_ransform(opt)

    def __getitem__(self, item):

        if self.opt.mode == 'Train':

            img_path = os.path.join(self.opt.Train_img_dir, self.Train_img_List[item])
            label_path = os.path.join(self.opt.Train_label_dir, self.Train_label_List[item])

            img = Image.open(img_path)
            label = Image.open(label_path)

            n_flip = random.random()
            if n_flip > 0.5:
                img = F.hflip(img)
                label = F.hflip(label)

            img = img.resize((256, 256), Image.ANTIALIAS)
            label = label.resize((256, 256), Image.ANTIALIAS)
            img = self.transform(img)
            label = self.transform(label)

            sample = {'img': img, 'img_path': self.Train_img_List[item],
                      'label': label, 'label_path': self.Train_label_List[item]}


        elif self.opt.mode == 'Test':
            img_path = os.path.join(self.opt.Test_img_dir, self.Test_img_List[item])
            label_path = os.path.join(self.opt.Test_label_dir, self.Test_label_List[item])

            img = Image.open(img_path)
            label = Image.open(label_path)

            n_flip = random.random()

            img = img.resize((256, 256), Image.ANTIALIAS)
            label = label.resize((256, 256), Image.ANTIALIAS)
            img = self.transform(img)
            label = self.transform(label)

            sample = {'img': img, 'img_path': self.Test_img_List[item],
                      'label': label, 'label_path': self.Test_label_List[item]}

        return sample

    def __len__(self):
        if self.opt.mode == 'Train':
            return len(self.Train_img_List)
        elif self.opt.mode == 'Test':
            return len(self.Test_img_List)


def BER(y_actual, y_hat):
    y_hat = torch.sigmoid(y_hat).ge(0.5).float()

    y_actual = y_actual.squeeze(1)
    y_hat = y_hat.squeeze(1)

    #output==1
    pred_p=y_hat.eq(1).float()

    #output==0
    pred_n = y_hat.eq(0).float()

    #TP
    tp_mat = torch.eq(pred_p,y_actual)
    TP = float(tp_mat.sum())

    #FN
    fn_mat = torch.eq(pred_n, y_actual)
    TN = float(fn_mat.sum())

    # FP
    fp_mat = torch.ne(y_actual, pred_p)
    FP = float(fp_mat.sum())

    # FN
    fn_mat = torch.ne(y_actual, pred_n)
    FN = float(fn_mat.sum())



    #print(TP,TN,FP,FN)
    #tot=TP+TN+FP+FN
    #print(tot)
    pos = TP+FN
    neg = TN+FP

    #print(pos,neg)

    #print(TP/pos)
    #print(TN/neg)
    if(pos!=0 and neg!=0):
        BAC = (.5 * ((TP / pos) + (TN / neg)))
    elif(neg==0):
        BAC = (.5*(TP/pos))
    elif(pos==0):
        BAC = (.5 * (TN / neg))
    else:
        BAC = .5

    BER=(1-BAC)
    return BER


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    opt = parser.parse_args()
    opt.Train_img_dir = '/home/media/Shadow_Detection/SBU-shadow/SBU-shadow/SBUTrain4KRecoveredSmall/ShadowImages'
    opt.Train_label_dir = '/home/media/Shadow_Detection/SBU-shadow/SBU-shadow/SBUTrain4KRecoveredSmall/ShadowMasks'
    opt.Test_img_dir = '/home/media/Shadow_Detection/SBU-shadow/SBU-shadow/SBU-Test/ShadowImages'
    opt.Test_label_dir = '/home/media/Shadow_Detection/SBU-shadow/SBU-shadow/SBU-Test/ShadowMasks'
    opt.mode = 'Train'
    opt.Train = True
    opt.shuffle = True
    opt.nThreads = 4
    opt.batchsize = 16
    dataset_shadow = CreateDataset_shadow(opt)
    dataloader_shadow = data.DataLoader(dataset_shadow, batch_size=opt.batchsize, shuffle=opt.shuffle,
                                        sampler=None, batch_sampler=None,
                                         num_workers=int(opt.nThreads))

    #optimizer = optim.SGD(ShadowNet.parameters() , lr=0.001, momentum=0.9)
    model = ShadowNet().to(device)
    #loss = nn.CrossEntropyLoss()
    loss=nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    loss_buffer = []
    eval_buffer = 1

    for epoch in range(3000):


        for i, data in enumerate(dataloader_shadow, 0):
            #get the inputs
            input = data['img'].to(device)
            label= data['label'].to(device)


            optimizer.zero_grad()

            output = model(input)

            ploss = loss(output,label)
            ploss.backward()
            optimizer.step()

            if i % 5 == 0:
                eval = BER(label,output)
                print('[%d, %5d] loss: %.3f eval: %.3f' %
                      (epoch + 1, i + 1, ploss, eval))

                if eval < eval_buffer:
                    torch.save(model.state_dict(), 'model_Best.pth')
                fig = plt.figure()
                plt.plot(loss_buffer)
                fig.savefig('Loss_plot.png')
                plt.close()
                eval_buffer = eval





    opt.mode = 'Test'
    opt.Test = True
    for epoch in range(3000):

        with torch.no_grad():
            for i, data in enumerate(dataloader_shadow, 0):
                # get the inputs
                input = data['img']
                label = data['label']

                output = model(input)

                teval = BER(label, output)

                output = torch.sigmoid(output)

                if i % 10 == 0:
                    print('[%d, %5d] teval: %.3f' %
                          (epoch + 1, i + 1, teval))

                    torchvision.utils.save_image(output, '[%d %d]out.jpg' % (epoch + 1, i + 1), normalize=True)
                    torchvision.utils.save_image(label, '[%d %d]label.jpg' % (epoch + 1, i + 1), normalize=True)
                    







