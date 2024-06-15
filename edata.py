import nibabel as nib
import matplotlib.pyplot as plt
import os

'''img = nib.load("segtraindata/Patient-42/42-LesionSeg-Flair.nii").get_fdata()
img = img[:, :, img.shape[2]//2]
plt.imshow(img, cmap='gray')
plt.show()'''

def nii_to_img(nii_path):
    img = nib.load(nii_path).get_fdata()
    img = img[:, :, img.shape[2]//2]
    return img


print(os.listdir('segtraindata'))

for folder in os.listdir('segtraindata'):
    path = 'segtraindata/'+folder
    i = int(folder[8:])
    img = nii_to_img(path+'/'+str(i)+'-LesionSeg-Flair.nii')
    plt.imsave(os.path.join('segtraindata3',str(i)+'.png'), img,cmap='gray')
