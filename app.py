from ultralytics import YOLO
import cv2
import torch
import numpy as np
from pathlib import Path

def load_model(weights_path):
    model = YOLO(weights_path)
    return model

def run_inference(model, image_path, output_path):
    # Run inference
    results = model(image_path)
    
    # Get the first result (assuming single image input)
    result = results[0]
    
    # Load the original image
    img = cv2.imread(image_path)
    
    # Draw bounding boxes
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        conf = box.conf[0]
        cls = box.cls[0]
        label = f'{result.names[int(cls)]} {conf:.2f}'
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(img, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
    
    # Save the output image
    cv2.imwrite(output_path, img)
    print(f"Output image saved to {output_path}")

if __name__ == "__main__":
    weights_path = "yolov10m.pt"
    image_path = "Inference-Images/human.jpg"
    output_path = "Inference-Images/human-result.jpg"
    
    # Load the model
    model = load_model(weights_path)
    
    # Run inference
    run_inference(model, image_path, output_path)