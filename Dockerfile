FROM python:3.9-slim

WORKDIR /app

COPY app.py .
COPY Inference.py .
COPY flower_model_complete.pth .

RUN pip install flask torch torchvision pillow

EXPOSE 5000

CMD ["python", "app.py"]
