from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import numpy as np
import cv2

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO("best.pt")

@app.get("/")
def home():
    return { "status": "ok"}
@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    data = await file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code= 400, detail="Image Not Found")
    results = model(img)[0]
    detections = []
    for box in results.boxes:
        detections.append({
            "class": model.names[int(box.cls)],
            "confidence" : round(float(box.conf), 2),
            "bbox" : [round(v, 1) for v in box.xyxy[0].tolist()],


        })
    return {"count": len(detections), "detections": detections}