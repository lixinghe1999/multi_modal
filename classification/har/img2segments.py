# Load all the images and save as a video segment with 20 frames.
import cv2
import os
import clip
import torch
from PIL import Image
import numpy as np

if __name__ == "__main__":
    
    torch.cuda.set_device(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model, preprocess = clip.load("ViT-B/32", device=device)

    dict_start_number = {'01':0,'02':0,'03':0,'04':0,'05':0,'06':0,
                         '07':0,'08':0,'09':0,'10':0,'11':0,'12':0,
                         '13':0,'14':0,'15':0,'16':0,'17':0,'18':0,
                         '19':0,'20':0,'21':0,'22':0,'23':0,'24':0,
                         '26':0,'27':0,'28':0,'29':0,'30':0}
    classnames = [
            'stand in a trance',
            'stand and play mobilephone',
            'squat',
            'rummage cabinet',
            'sit on chair in a trance',
            'sleep'
            ]
    class_to_label = {'06':0,'08':1,'15':2,'20':3,'22':4,'29':5}
    
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    text = torch.cat([clip.tokenize(f"a photo of a person doing {c}.") for c in classnames]).to(device)


    corr = 0
    total = 0
    for subject in range(2,32):
        if subject<10:
            subject_idx = '0'+str(subject)
        else:
            subject_idx = str(subject)
        root_base = 'RGB/v'+subject_idx+'/'


        for cls in class_to_label:
            path_read_root = root_base+cls
            path_save_root = 'Video_rgb_jpg/'+'a'+cls
            path_save_root_feat = 'Video_feat/'+'a'+cls
            path_save_root_img = 'Video_rgb/'+'a'+cls

            frame_names = os.listdir(path_read_root)
            frame_names.sort()
            
            idx = 0
            sample_idx = dict_start_number[cls]
            while idx+16 < len(frame_names):
                frame_names_seg = frame_names[idx:idx+16]

                sample_features = torch.zeros(len(frame_names_seg),512)
                sample_img = torch.zeros(len(frame_names_seg),3,224,224)
                for ii in range(len(frame_names_seg)):
                    # 1. Save CLIP features 
                    im = preprocess(Image.open(os.path.join(path_read_root,frame_names_seg[ii]))).unsqueeze(0).to(device)
                    # Extract clip features for each frame
                    with torch.no_grad():
                        im_features = model.encode_image(im)

                        logits_per_image, logits_per_text = model(im, text)
                        probs = logits_per_image.softmax(dim=-1)
                    _, pred = torch.max(probs, dim=1)
                    if pred == class_to_label[cls]:
                        corr += 1 
                    total += 1
                    
                      
                    sample_features[ii,:] = im_features
                    sample_img[ii,:,:,:] = im
                    # im2 = im[0,:,:,:].cpu().numpy().transpose(1, 2, 0)
                    
                    # # 2. Load and save origin image
                    # im = cv2.imread(os.path.join(path_read_root,frame_names_seg[ii]))
                    # cv2.imwrite(os.path.join(dir_sv,frame_names_seg[ii]), im)
                
                # Save CLIP features (20-frames)
                if not os.path.exists(path_save_root_feat):
                    os.makedirs(path_save_root_feat)
                filename = os.path.join(path_save_root_feat,'a'+cls+'_v'+subject_idx+'_'+frame_names_seg[ii][0:-4]+'.npy')
                np.save(filename,sample_features.cpu().numpy())
                
                
                # Save 3x224x224 sample (20-frames)
                if not os.path.exists(path_save_root_img):
                    os.makedirs(path_save_root_img)
                filename = os.path.join(path_save_root_img,'a'+cls+'_v'+subject_idx+'_'+frame_names_seg[ii][0:-4]+'.npy')
                np.save(filename,sample_img.cpu().numpy())
                
                idx += 12 # slide 12 frames
                sample_idx += 1
                
            dict_start_number[cls] = sample_idx

    print(corr/total)