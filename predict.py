from ultralytics import YOLO

model = YOLO('best.pt')

results = model.predict("test2.jpg", conf = 0.25 , save = True)

for box in results[0].boxes:
    print(model.names[int(box.cls)], round(float(box.conf), 2))

