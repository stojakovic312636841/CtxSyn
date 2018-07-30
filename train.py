import sys, os
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm

from model import GridNet
from LapLoss import LapLoss

from ptsemseg.loader import get_loader, get_data_path
from ptsemseg.metrics import runningScore
from ptsemseg.loss import *
from ptsemseg.augmentations import *

import pdb

def train(args):

    #data_aug = Compose([RandomRotate(10),RandomHorizontallyFlip()])
    data_aug = None

    # Setup dataloader
    data_loader = get_loader(args.dataset)
    # data_path = get_data_path(args.dataset)
    data_path = args.dataset_dir
    t_loader = data_loader(data_path, is_transform = True,
                           img_size = (args.img_rows, args.img_cols),
                           augmentations = data_aug, img_norm = args.img_norm)
    v_loader = data_loader(data_path, is_transform = True,
                           split = 'validation', img_size = (args.img_rows, args.img_cols),
                           img_norm = args.img_norm)
    
    n_classes = t_loader.n_classes
    trainloader = data.DataLoader(t_loader, batch_size = args.batch_size, shuffle = True)
                                  # num_workers = args.num_workers, shuffle = True)
    valloader = data.DataLoader(v_loader, batch_size = args.batch_size)
                                # num_workers = args.num_workers)
    
    # Setup metrics
    running_metrics = runningScore(n_classes)

    # Setup visdom for visualization
    if args.visdom:
        vis = visdom.Visdom()
        loss_window = vis.line(X = torch.zeros((1)).cpu(),
                               Y = torch.zeros((1)).cpu(),
                               opts = dict(xlabel = 'minibatches',
                                           ylabel = 'Loss',
                                           title = 'Training loss',
                                           legend = ['Loss']))

    gpu_id = int(input('input utilize gpu id (-1:cpu) : '))
    #device = torch.cuda.device(gpu_id if gpu_id >= 0 else 'cpu')
    print("cuda device is %d"%(gpu_id))  

    # Setup model
    model = GridNet(in_chs = 134, out_chs = n_classes)#3
    #model.to(device)
    if gpu_id >=0:
       model.cuda(gpu_id)
    else:
       model.cpu()
        
    if hasattr(model.modules, 'optimizer'):
        optimizer = model.modules.optimizer
    else:
        #optimizer = torch.optim.SGD(model.parameters(), lr = args.l_rate,momentum = 0.99, weight_decay = 5e-4)
        optimizer = torch.optim.Adamax(model.parameters(), lr = args.l_rate,betas=(0.9,0.999))

	#criterion = nn.NLLLoss()
    if args.loss_function == 'L1_loss':
       criterion = nn.L1Loss()
    elif args.loss_function == 'VGG_loss':
       criterion = nn.NLLLoss()
    elif args.loss_function == 'Lap_loss':
       criterion = LapLoss(max_levels=5,gpu_id = gpu_id)

    if gpu_id >=0:
       criterion.cuda(gpu_id)
    else:
       criterion.cpu()

    if args.resume is not None:
        if os.path.isfile(args.resume):
            print('Loading model and optimizer from checkpoint {%s}'%(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            print("Loaded checkpoint {%s}, epoch {%s}"%(args.resume,checkpoint['epoch']))
        else:
            print('No checkpoint found at {%s}'%(args.resume))

    best_iou = 99999.0
    for epoch in range(args.n_epoch):
        print('epoch : %d start'%(epoch))
        
        model.train()

        for i, (images, labels) in enumerate(trainloader):
            print('epoch : %d, num_batch : %d processing...'%(epoch,i))
            images, labels = images.cuda(gpu_id), labels.cuda(gpu_id)
            labels = Variable(labels)

			# zero the gradient buffers
            optimizer.zero_grad()
            #print('infering...')
			#train_forward
            outputs = model.forward(Variable(images))

            # pdb.set_trace()
            #print('loss calculating')
            loss = criterion(outputs, labels)
        
            #print('back propagating')
            loss.backward()
            #print('parameter update')
            optimizer.step()

            if args.visdom:
                vis.line(X = torch.ones((1, 1)).cpu() * i,
                         Y = torch.Tensor([loss.data[0]]).unsqueeze(0).cpu(),
                         win = loss_window,
                         update = 'append')

            if (i+1) % 1 == 0:
                print("Epoch [%d/%d] Loss: %f"%(epoch+1,args.n_epoch,loss.cpu().data.numpy()))

            model.eval()
            #print('model eval')
            eval_loss_sum = 0.0
            for i_val, (images_val, labels_val) in tqdm(enumerate(valloader)):
                images_val, labels_val = images_val.cuda(gpu_id), labels_val.cuda(gpu_id)
                labels_val = Variable(labels_val)

                eval_outputs = model(Variable(images_val))
                #pred = outputs.data.max(1)[1].cpu().numpy()
                #gt = labels_val.data.cpu().numpy()
                loss_val = criterion(eval_outputs, labels_val)   
                eval_loss_sum += loss_val.cpu().data.numpy()

            if eval_loss_sum <= best_iou:
                best_iou = eval_loss_sum
                state = {'epoch': epoch+1,
                         'model_state': model.state_dict(),
                         'optimizer_state': optimizer.state_dict()}
                torch.save(state, '%s_best_model.pkl'%(args.dataset))
                print("Epoch [%d/%d] Eval_Loss: %f --> save model"%(epoch+1,args.n_epoch,eval_loss_sum)) 
  


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--dataset', nargs='?', type=str, default='mydata',
                        help = 'Dataset to use [\'pascal, camvid, ade20k etc\']')#default='mit_sceneparsing_benchmark'
    parser.add_argument('--dataset_dir', required = True, type = str,default='./',
                        help = 'Directory containing target dataset')
    parser.add_argument('--img_rows', nargs='?', type=int, default=256,
                        help = 'Height of the input')
    parser.add_argument('--img_cols', nargs='?', type=int, default=256,
                        help = 'Width of input')

    parser.add_argument('--img_norm', dest = 'img_norm', action = 'store_true',
                        help = 'Enable input images scales normalization [0, 1] | True by default')
    parser.add_argument('--no-img_norm', dest = 'img_norm', action = 'store_false',
                        help = 'Disable input images scales normalization [0, 1] | True by Default')
    parser.set_defaults(img_norm = True)
    
    parser.add_argument('--n_epoch', nargs = '?', type = int, default = 30,
                        help = '# of epochs')
    parser.add_argument('--batch_size', nargs = '?', type = int, default = 4,
                        help = 'Batch size')
    parser.add_argument('--l_rate', nargs = '?', type = float, default = 0.001,
                        help = 'Learning rate [1-e5]')
    parser.add_argument('--resume', nargs = '?', type = str, default = None,
                        help = 'Path to previous saved model to restart from')
    parser.add_argument('--loss_function', nargs = '?', type = str, default = 'L1_loss',
                        help = 'consider various loss functions that measure the difference between the interpolated frame and its GT')
    parser.add_argument('--visdom', dest = 'visdom', action = 'store_true',
                        help = 'Enable visualizaion(s) on visdom | False by default')
    parser.add_argument('--no-visdom', dest = 'visdom', action = 'store_false',
                        help = 'Disable visualization(s) in visdom | False by default')
    parser.set_defaults(visdom = False)

    args = parser.parse_args()
    for key, item in vars(args).items():
       print('%s : %s'%(key,item))
    train(args)

























                    
