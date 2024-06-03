from ultralytics import YOLO

model = YOLO('yolomodel.pt')

model.predict(source="All Data\MS\MS_14.png",show=True, conf=0.4, save=True) # substitute the image location

input("Press enter to exit.")