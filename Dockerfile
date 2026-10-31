FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --default-timeout=300 --retries 10 torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=300 --retries 10 -r requirements.txt

COPY app.py .
#COPY yolo11n.pt .
COPY best.pt .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

