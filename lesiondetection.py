from ultralytics import YOLO

model = YOLO('yolomodel.pt')

model.predict(source="C:/Users/osama/Pictures/Research/ADEM/ADEM_11.jpg",show=True)

input()