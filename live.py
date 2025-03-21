from ultralytics import YOLO
import cv2
import torch
import numpy as np
from pathlib import Path
import argparse

def load_model(weights_path):
    model = YOLO(weights_path)
    return model

def run_inference_on_image(model, image_path, output_path):
    """Run inference on a single image and save the output."""
    # Run inference
    results = model(image_path)
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

def run_inference_on_video(model, video_path, output_path):
    """Run inference on a pre-recorded video and save the output."""
    video_capture = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30, (int(video_capture.get(3)), int(video_capture.get(4))))

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Run inference
        results = model(frame)
        result = results[0]

        # Draw bounding boxes
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = box.conf[0]
            cls = box.cls[0]
            label = f'{result.names[int(cls)]} {conf:.2f}'
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)

        out.write(frame)

    video_capture.release()
    out.release()
    print(f"Output video saved to {output_path}")

def run_inference_on_webcam(model, output_path=None):
    """Run inference on a live USB webcam feed."""
    video_capture = cv2.VideoCapture(0)

    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 30, (int(video_capture.get(3)), int(video_capture.get(4))))

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Run inference
        results = model(frame)
        result = results[0]

        # Draw bounding boxes
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = box.conf[0]
            cls = box.cls[0]
            label = f'{result.names[int(cls)]} {conf:.2f}'
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)

        if output_path:
            out.write(frame)

        cv2.imshow('Object Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run object detection on image, video, or webcam')
    parser.add_argument('--weights', required=True, help='Path to the YOLO weights file')
    parser.add_argument('--input-type', choices=['image', 'video', 'webcam'], required=True, help='Input type (image, video, or webcam)')
    parser.add_argument('--input-path', help='Path to the input image or video file')
    parser.add_argument('--output-path', help='Path to save the output image or video')
    args = parser.parse_args()

    # Load the model
    model = load_model(args.weights)

    if args.input_type == 'image':
        run_inference_on_image(model, args.input_path, args.output_path)
    elif args.input_type == 'video':
        run_inference_on_video(model, args.input_path, args.output_path)
    elif args.input_type == 'webcam':
        run_inference_on_webcam(model, args.output_path)