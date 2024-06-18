from ultralytics import YOLO
import cv2

model = YOLO('segmentationmodel.pt')

model.predict('All Data/MS/MS_10.png',show=True)

cv2.waitKey()