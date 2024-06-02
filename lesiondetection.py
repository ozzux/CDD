from ultralytics import YOLO

model = YOLO('yolomodel.pt')

model.predict(source="IMG LOCATION",show=True) # substitute the image location

input()