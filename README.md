# Steel Surface Defect Detection API

A YOLO11 model fine-tuned to detect three types of defects on steel surfaces, served as a REST API with FastAPI and packaged in Docker. Upload an image, get back the detected defects as JSON.

## Classes
- `crescent_gap`
- `punching_hole`
- `welding_line`

## Tech stack
Python, Ultralytics YOLO11, FastAPI, Uvicorn, OpenCV, Docker, Google Colab (T4 GPU) for training

## Results (test set, 71 images never seen during training)

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| **all** | 0.889 | 0.928 | **0.933** | **0.625** |
| crescent_gap | 1.000 | 0.991 | 0.995 | 0.857 |
| punching_hole | 0.950 | 1.000 | 0.995 | 0.668 |
| welding_line | 0.717 | 0.793 | 0.810 | 0.350 |

mAP was computed with Ultralytics' default settings (no custom confidence threshold). Precision and recall are reported by Ultralytics at its own operating point, not at a fixed 0.25 threshold. On CPU (Intel Core i5-1235U) Ultralytics reported about 70 ms inference per image during this evaluation.

### What the numbers say
- `crescent_gap` and `punching_hole` are detected almost perfectly.
- `welding_line` is the weak class. It is a thin, very long line, and the predicted box often covers only part of the seam or is split into two boxes. This shows up as a low mAP50-95 (0.350): the line is usually found, but the box is not tight enough.

## Training details
- Base model: pretrained `yolo11n.pt`, fine-tuned for 50 epochs at 640 px
- Data split: 496 train, 142 validation, 71 test images (Roboflow split, no Roboflow augmentation)
- Ultralytics default augmentations were active during training
- I also tried `imgsz=960`. Validation mAP50-95 moved from 0.632 to 0.645, a small difference from a single run, and I did not evaluate it on the test set, so I cannot say it is a real improvement. The model in this repo is the 640 px one.

## Limitations
- Small dataset (709 images), and the test set has only 71 images, so numbers can move by a few points between runs.
- Results come from one training run per setting.
- `welding_line` needs work. Ideas I have not tried yet: a larger model (`yolo11s`), higher resolution, less mosaic augmentation, and checking whether the dataset's `welding_line` boxes are labelled consistently.

## Run with Docker
```bash
docker build -t steel-defect-api .
docker run -d -p 8000:8000 --name steel-api steel-defect-api
```
Open http://127.0.0.1:8000/docs to try the API.

## Run locally
```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install ultralytics fastapi uvicorn python-multipart
uvicorn app:app --reload
```

## API example
`POST /detect` with an image file (form field name: `file`). Example response for `test2.jpg`:
```json
{
  "count": 2,
  "detections": [
    {"class": "punching_hole", "confidence": 0.83, "bbox": [93.6, 555.9, 188.1, 611.9]},
    {"class": "welding_line", "confidence": 0.48, "bbox": [230.7, 379.1, 1765.9, 433.4]}
  ]
}
```
`bbox` is `[x1, y1, x2, y2]` in pixels of the original image.

## Project structure
- `app.py`: FastAPI app
- `Dockerfile`, `requirements.txt`: container setup
- `best.pt`: trained weights
- `predict.py`: run the model on a single image
- `evaluate.py`: evaluate on the test set (needs the dataset in `dataset/`)

## Dataset
Steel surface defect dataset on Roboflow Universe, by "AI project", licensed CC BY 4.0:
https://universe.roboflow.com/object-detection-vka4o/steel-surface-defect-ejtjk-aq7yz/dataset/1

The dataset is not included in this repo. To reproduce the evaluation, download it in YOLOv8 format and place it in `dataset/`.
