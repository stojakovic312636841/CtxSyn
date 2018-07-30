import sys, os
import torch
import scipy.io
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import matplotlib.pyplot as plt

from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm
from random import shuffle

from model import GridNet
from LapLoss import LapLoss

from ptsemseg.loader import get_loader, get_data_path
from ptsemseg.metrics import runningScore
from ptsemseg.loss import *
from ptsemseg.augmentations import *



def make_dataset(root, train=False):
    eval_dataset = []
	
    pathDir = sorted(os.listdir(root))
    #print(sorted(pathDir))
    
    if not train:
       for allDir in pathDir:
           child_path = os.path.join(root,allDir)
           flow_name = 'frames_flow.mat'
           groundTruth_name = 'ground_truth.mat'
           feature_name = 'features.mat'
           eval_dataset.append([os.path.join(child_path,flow_name),os.path.join(child_path,feature_name),os.path.join(child_path,groundTruth_name)])
           
    shuffle(eval_dataset)
    print(eval_dataset[0])

    return eval_dataset


def transform(img):
    """transform

    :param img:
    """
    img = img.astype(np.float64)#[:,:,::-1]  

    # NHWC -> NCHW
    img = img.transpose(2, 0, 1)            
    img = torch.from_numpy(img).float().unsqueeze(0)

    return img


def reverse_trans(img,tran=True):
    #img = img * 255.0
    img[img < 0] = 0
    #img[img > 255] = 255.0
    img[img > 1] = 1.0
    if tran:
       img = img.transpose(1, 2, 0) 

    return img


def image_show(image,num,savedir):
    plt.figure(num)
    plt.imshow(image)
    plt.axis('off')
    if num==2:
       plt.title('IF')
       plt.savefig(savedir+'result_IF.png')
    if num==1:
       plt.title('GT')
       plt.savefig(savedir+'result_gt.png')


def demo(args):
    eval_list = make_dataset(args.input_dir)
    
    model = GridNet(in_chs = 134, out_chs = 3)
    model.cuda(1)

    if args.model_path is not None:
        if os.path.isfile(args.model_path):
            print('Loading model and optimizer from checkpoint {%s}'%(args.model_path))
            checkpoint = torch.load(args.model_path)
            model.load_state_dict(checkpoint['model_state'])
            print("Loaded checkpoint {%s}, epoch {%s}"%(args.model_path,checkpoint['epoch']))
        else:
            print('No checkpoint found at {%s}'%(args.model_path))

    for item  in eval_list:
        flow_path = item[0]
        feature_path = item[1]
        label_path = item[2]

        flow = scipy.io.loadmat(flow_path)
        feature = scipy.io.loadmat(feature_path)
        label = scipy.io.loadmat(label_path)['label']

        #concate 4 picture
        tup = (flow['frame_forward'],flow['frame_backward'],feature['feature_next'],feature['feature_prev'])
        img = np.concatenate(tup, axis=2)
        img = transform(img)
        img = img.cuda(1)
        
        outputs = model(Variable(img))

        outputs = outputs.cpu().data.numpy().squeeze(0)
        outputs = reverse_trans(outputs)        
        #outputs = outputs.transpose(1, 2, 0)
 
        #label = reverse_trans(label,tran=False)
        image_show(label,1,args.out)
        image_show(outputs,2,args.out)   
        plt.show()     
        #print(outputs[:,:,0],outputs[:,:,1],outputs[:,:,2])
        os._exit(0)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')

    '''
    parser.add_argument('--first', required = True, type = str,default='',
                        help = 'Directory containing the first frame')
    parser.add_argument('--second', required = True, type = str,default='',
                        help = 'Directory containing the second frame')
    '''
    parser.add_argument('--input_dir', required = True, type = str,default='',
                        help = 'Directory containing all of the infomation')
    parser.add_argument('--model_path', required = True, type = str,default='',
                        help = 'Directory containing the model')
    parser.add_argument('--out', nargs = '?', type = str, default = './',
                        help = 'Directory containing the output frame')
    parser.add_argument('--visdom', dest = 'visdom', action = 'store_true',
                        help = 'Enable visualizaion(s) on visdom | False by default')
    parser.add_argument('--no-visdom', dest = 'visdom', action = 'store_false',
                        help = 'Disable visualization(s) in visdom | False by default')
    parser.set_defaults(visdom = False)

    args = parser.parse_args()
    for key, item in vars(args).items():
       print('%s : %s'%(key,item))
    demo(args)
