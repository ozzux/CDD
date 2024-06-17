import nibabel as nib
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2


for patient in os.listdir('train'):
    num = int(patient[8:])
    print(str(num))
    img1 = nib.load('train/Patient-'+str(num)+'/'+str(num)+'-LesionSeg-Flair.nii').get_fdata()
    img1 = img1[:, :, img1.shape[2]//2]

    img2 = nib.load('train/Patient-'+str(num)+'/'+str(num)+'-Flair.nii').get_fdata()
    img2 = img2[:, :, img2.shape[2]//2]

    plt.imsave('traindata/images/'+str(num)+'.jpg', img2, cmap='gray')
    plt.imsave('traindata/masks/'+str(num)+'.jpg', img1, cmap='gray')



