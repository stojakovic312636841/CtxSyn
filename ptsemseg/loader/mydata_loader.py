import os,glob,random
import torch
import numpy as np
import scipy.misc as m
import scipy.io

from random import shuffle
from torch.utils import data

from ptsemseg.utils import recursive_glob


def make_dataset(root, train=True):
    train_dataset = []
    validation_dataset = []
	
    pathDir = sorted(os.listdir(root))
    #print(sorted(pathDir))
    
    if train:
       for allDir in pathDir:
           child_path = os.path.join(root,allDir)
           flow_name = 'frames_flow.mat'
           groundTruth_name = 'ground_truth.mat'
           feature_name = 'features.mat'
           train_dataset.append([os.path.join(child_path,flow_name),os.path.join(child_path,feature_name),os.path.join(child_path,groundTruth_name)])

    length=len(train_dataset)
    train_dataset , validation_dataset = train_dataset[:int(length//10)*(-1)],train_dataset[int(length//10)*(-1):]

    return train_dataset,validation_dataset



class mydataLoader(data.Dataset):
    """mydataLoader

    http://sceneparsing.csail.mit.edu/

    Data is derived from ADE20k, and can be downloaded from here:
    http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip

    NOTE: this loader is not designed to work with the original ADE20k dataset;
    for that you will need the ADE20kLoader

    This class can also be extended to load data for places challenge:
    https://github.com/CSAILVision/placeschallenge/tree/master/sceneparsing

    """
    def __init__(self, root, split="training", is_transform=False, img_size=512, augmentations=None, img_norm=True):
        """__init__

        :param root:
        :param split:
        :param is_transform:
        :param img_size:
        """
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.img_norm = img_norm
        self.n_classes = 3  # 0 is reserved for "other"
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.mean = np.array([104.00699, 116.66877, 122.67892])
        self.files = {}
        train_dataset,valid_dataset = make_dataset(self.root)
        if self.split == 'training':
           self.files[split] = train_dataset
        elif self.split == 'validation':
           self.files[split] = valid_dataset
        '''
        self.images_base = os.path.join(self.root, 'images', self.split)
        print('self.images_base:{%s}'%(self.images_base))
        self.annotations_base = os.path.join(self.root, 'annotations', self.split)
        print('self.annotations_base:{%s}'%(self.annotations_base))
        self.files[split] = recursive_glob(rootdir=self.images_base, suffix='.bmp')
        '''
        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        print("Found %d %s images" % (len(self.files[split]), split))
        


    def __len__(self):
        """__len__"""
        return len(self.files[self.split])



    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        flow_path = self.files[self.split][index][0]
        feature_path = self.files[self.split][index][1]
        label_path = self.files[self.split][index][2]
        
        #print('flow_path = %s'%(flow_path))
        #print('feature_path = %s'%(feature_path))
        #print('label_path = %s'%(label_path))
        
        flow = scipy.io.loadmat(flow_path)
        #flow_frame_forward = flow['frame_forward']
        #flow_frame_backward = flow['frame_backward']
        
        feature = scipy.io.loadmat(feature_path)
        #feature_next = feature['feature_next']
        #feature_prev = feature['feature_prev']

        label = scipy.io.loadmat(label_path)
        lbl = label['label']

        #concate 4 picture
        tup = (flow['frame_forward'],flow['frame_backward'],feature['feature_next'],feature['feature_prev'])
        img = np.concatenate(tup, axis=2)

        #crop the img and label into 256*256 patch
        bias = random.randint(1,40)
        img = img[bias:bias+self.img_size[0],bias:bias+self.img_size[1],:]
        lbl = lbl[bias:bias+self.img_size[0],bias:bias+self.img_size[1],:]
        #print(img.shape,lbl.shape)

        if self.augmentations is not None:
            img, lbl = self.augmentations(img, lbl)
        
        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        return img, lbl



    def transform(self, img, lbl):
        """transform

        :param img:
        :param lbl:
        """
        #img = m.imresize(img, (self.img_size[0], self.img_size[1])) # uint8 with RGB mode
        #img = img[:, :, ::-1] # RGB -> BGR
        img = img.astype(np.float64)
        #img -= self.mean
        if self.img_norm:
            # Resize scales images from 0 to 255, thus we need
            # to divide by 255.0
            #img = img.astype(float) / 255.0    
            pass        
        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)        
        
        lbl = lbl.astype(float)
        if self.img_norm:
           #lbl = lbl/255.0
           pass
        #lbl = m.imresize(lbl, (self.img_size[0], self.img_size[1]), 'nearest', mode='F')
        #lbl = lbl.astype(int)

        # NHWC -> NCHW
        lbl = lbl.transpose(2, 0, 1) 
        '''
        classes = np.unique(lbl)
        if not np.all(classes == np.unique(lbl)):
            print("WARN: resizing labels yielded fewer classes")
        
        if not np.all(np.unique(lbl) < self.n_classes):
            raise ValueError("Segmentation map contained invalid class values")
        '''      
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).float()

        return img, lbl
