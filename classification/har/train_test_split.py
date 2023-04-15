# Split train and test set.
import cv2
import os
import clip
import torch
from PIL import Image
import numpy as np
import sys
sys.path.append('/home/bufang/DepthCLIP/')

def save_list2text(filename, save_list):
    with open(filename,'w') as f:
        for i in save_list:
            for j in i:
                f.write(str(j))
            f.write('\n')
        f.close()


if __name__ == "__main__":
    
    torch.cuda.set_device(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    root_base = 'Dataset/Video_feat'
    cls_names = os.listdir(root_base)
    
    train_names,test_names = [],[]
    for cls in cls_names:
        folder = os.path.join(root_base, cls)
        sample_names = os.listdir(folder)
        sample_names.sort() # subject-independent split
        N_SAMPLE = len(sample_names)
        TRAIN_RATIO = 0.5
        # a = 297  
        train_names_cls = sample_names[0:round(N_SAMPLE*TRAIN_RATIO)]
        test_names_cls = sample_names[round(N_SAMPLE*TRAIN_RATIO):]
        
        for names in train_names_cls:
            train_names.append(os.path.join(folder,names))

        for names in test_names_cls:
            test_names.append(os.path.join(folder,names))
    
        save_list2text('Dataset/train_split_feat.txt',train_names)
        save_list2text('Dataset/test_split_feat.txt',test_names)