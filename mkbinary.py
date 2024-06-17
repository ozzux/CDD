import cv2 
import os
  
# read the image file 
for i in range(len(os.listdir('traindata/masks'))):
    img = cv2.imread('traindata/masks/'+str(i+1)+'.jpg',2) 
    
    ret, bw_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY) 
    
    # converting to its binary form 
    bw = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY) 
    
    cv2.imwrite('binary/'+str(i+1)+'.jpg', bw_img)