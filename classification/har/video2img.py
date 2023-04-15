# Load video data (avi) extract frames and save images
import cv2
import os
import torch
'''
This script will extract the useful data from AD-Box dataset
'''
if __name__ == "__main__":
    torch.cuda.set_device(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for subject in range(2,32):
        if subject<10:
            subject_idx = '0'+str(subject)
        else:
            subject_idx = str(subject)
        root_base = '../dataset/data_raw/v'+subject_idx+'/'

        CLASS_STATIC = ['01','02','03','04','05','06','07','08','09','10',
                '11','12','13','14','15','16','17','18','19','20',
                '21','22','23','24','26','27','28','29','30']
        
        for cls in CLASS_STATIC:
            path_read = root_base+'v'+subject_idx+'a'+cls+'/rgb.avi'
            path_save = 'RGB/'+'v'+subject_idx+'/'+cls
            
            # Read video and extract to frames
            vidcap = cv2.VideoCapture(path_read)
            success, image = vidcap.read()
            count = 0
            while success:
                if not count % 4:
                    if not os.path.exists(path_save):
                        os.makedirs(path_save)
                    if count < 10:
                        c = '000'+str(count)
                    elif count < 100:
                        c = '00'+str(count)
                    elif count < 1000:
                        c = '0'+str(count)
                    else:
                        c = str(count)
                    cv2.imwrite(path_save+"/frame"+c+".jpg", image)     # save frame as JPEG file      
                success,image = vidcap.read()
                print('Read a new frame: ', success)
                count += 1