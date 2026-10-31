from ultralytics import YOLO

model = YOLO("best.pt")
m = model.val(data="dataset/data.yaml", split="test", device="cpu")
print("mAP50:", m.box.map50, "mAP50-95:", m.box.map)
print("class-wise mAP50-95:", m.box.maps)