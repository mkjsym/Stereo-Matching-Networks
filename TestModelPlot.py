from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import skimage
import skimage.io
import skimage.transform
import numpy as np
import time
import math
import copy
from dataloader import KITTIloader2015 as ls
from dataloader import KITTILoader as DA
from models import *
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--maxdisp', type=int ,default=192,
                    help='maxium disparity')
parser.add_argument('--model', default='stackhourglass_bsconv',
                    help='select model')
parser.add_argument('--datatype', default='2012',
                    help='datapath')
parser.add_argument('--datapath', default=r'/home/youngmin/YM/SL_disk_b/datasets/KITTI/kitti_2012/data_stereo_flow/training/',
                    help='datapath')
parser.add_argument('--epochs', type=int, default=300,
                    help='number of epochs to train')
parser.add_argument('--loadmodel', default=r'/home/youngmin/YM/SL_disk_b/savemodels/stereo/kitti2012_bsconv_300/finetune_300.tar',
                    help='load model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.datatype == '2015':
   from dataloader import KITTIloader2015 as ls
elif args.datatype == '2012':
   from dataloader import KITTIloader2012 as ls

all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = ls.dataloader(args.datapath)

TestImgLoader = torch.utils.data.DataLoader(
         DA.myImageFloder(test_left_img,test_right_img,test_left_disp, False), 
         batch_size= 8, shuffle= False, num_workers= 4, drop_last=False)

if args.model == 'stackhourglass':
    model = stackhourglass(args.maxdisp)
elif args.model == 'stackhourglass_bsconv':
    model = stackhourglass_bsconv(args.maxdisp)
elif args.model == 'basic':
    model = basic(args.maxdisp)
else:
    print('no model')

if args.cuda:
    model = nn.DataParallel(model)
    model.cuda()

# if args.loadmodel is not None:
#     state_dict = torch.load(args.loadmodel)
#     model.load_state_dict(state_dict['state_dict'])

# print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

optimizer = optim.Adam(model.parameters(), lr=0.1, betas=(0.9, 0.999))

def test(imgL,imgR,disp_true):
        model.eval()
        imgL   = Variable(torch.FloatTensor(imgL))
        imgR   = Variable(torch.FloatTensor(imgR))   
        if args.cuda:
            imgL, imgR = imgL.cuda(), imgR.cuda()

        with torch.no_grad():
            output3 = model(imgL,imgR)

        pred_disp = output3.data.cpu()
        pred_disp = torch.squeeze(pred_disp, 1) #revised part
        
        #computing 3-px error#
        true_disp = copy.deepcopy(disp_true)
        index = np.argwhere(true_disp>0)
        disp_true[index[0][:], index[1][:], index[2][:]] = np.abs(
            true_disp[index[0][:], index[1][:], index[2][:]] - pred_disp[index[0][:], index[1][:], index[2][:]])
        correct = (disp_true[index[0][:], index[1][:], index[2][:]] < 3) | (
                disp_true[index[0][:], index[1][:], index[2][:]] < true_disp[
            index[0][:], index[1][:], index[2][:]]*0.05)      
        torch.cuda.empty_cache()

        return 1-(float(torch.sum(correct))/float(len(index[0])))

def main():
    total_test_loss = 0

    # if args.loadmodel is not None:
    #     state_dict = torch.load(args.loadmodel)
    #     model.load_state_dict(state_dict['state_dict'])

    # print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    
    x = np.array([])
    y = np.array([])

    savepath = r'/home/youngmin/YM/SL_disk_b/GitHub/Stereo-Matching-Networks/test_results/'
    path = r'/home/youngmin/YM/SL_disk_b/savemodels/stereo/'
    savedata = 'kitti2012_bsconv_1000'
    savedata_name = 'kitti2012_bsconv_300'
    path = path + savedata + '/finetune_'
    for i in range(10, 301, 10):
        total_test_loss = 0
        tmp = path + str(i) + '.tar'
        print(tmp)
        
        state_dict = torch.load(tmp)
        model.load_state_dict(state_dict['state_dict'])

        # print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

        ## Test ##
        for batch_idx, (imgL, imgR, disp_L) in enumerate(TestImgLoader):
            test_loss = test(imgL,imgR, disp_L)
            total_test_loss += test_loss
        print('total 3-px error in val = %.3f' %(total_test_loss/len(TestImgLoader)*100))

        x = np.append(x, [i])
        y = np.append(y, (total_test_loss/len(TestImgLoader)*100))
        # model = nn.DataParallel(stackhourglass(args.maxdisp))
    
    plt.plot(x, y)
    plt.xlabel('epochs')
    plt.ylabel('3-pixel error')
    plt.title('KITTI 2012')
    plt.savefig(savepath + savedata_name + '_noc.png')
    np.save(savepath + savedata_name + '_noc_x', x)
    np.save(savepath + savedata_name + '_noc_y', y)
    
if __name__ == '__main__':
   main()
