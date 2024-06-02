from ultralytics import YOLO

model = YOLO('yolomodel.pt')

model.predict(source="IMG SOURCE",show=True) # substitute the image location

input("Press enter to exit.")