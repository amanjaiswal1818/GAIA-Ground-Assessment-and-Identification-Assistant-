import cv2
import numpy as np
from pathlib import Path
import argparse
from queue import Queue
from threading import Thread
import onnxruntime as ort
import json
import sys

def load_model(weights_path):
    print(f"Loading model from {weights_path}")
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(weights_path, providers=providers)
    print("Model loaded successfully")
    return session

def load_class_names(file_path):
    print(f"Loading class names from {file_path}")
    file_extension = Path(file_path).suffix.lower()
    if file_extension == '.json':
        with open(file_path, 'r') as f:
            class_names = json.load(f)
    elif file_extension == '.txt':
        with open(file_path, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
    else:
        raise ValueError("Unsupported file format. Please use .json or .txt")
    print(f"Loaded {len(class_names)} class names")
    return class_names

def process_frame(session, frame, imgsz=640):
    print("Processing frame")
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    img = cv2.resize(frame, (imgsz, imgsz))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0

    results = session.run([output_name], {input_name: img})
    print("Frame processed")
    return results[0]

def draw_boxes(frame, result, class_names):
    print("Drawing bounding boxes")
    #print(result.shape)
    for detection in result[0]:
        #print('Check-1')
        if detection[4] > 0.2:  # Confidence threshold
            class_id = int(detection[5])
            confidence = detection[4]
            bbox = detection[:4]
            print(confidence) #Check-2
            x1, y1, x2, y2 = bbox
            x1 = int(x1 * frame.shape[1])
            y1 = int(y1 * frame.shape[0])
            x2 = int(x2 * frame.shape[1])
            y2 = int(y2 * frame.shape[0])
            print(bbox)
            label = f'{class_names[class_id]} {confidence:.2f}'
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
    print("Bounding boxes drawn")
    return frame

def run_inference_on_image(session, image_path, output_path, class_names, imgsz=640):
    print(f"Running inference on image: {image_path}")
    img = cv2.imread(image_path)
    result = process_frame(session, img, imgsz=imgsz)
    img = draw_boxes(img, result, class_names)
    cv2.imwrite(output_path, img)
    print(f"Output image saved to {output_path}")

def run_inference_on_video(session, video_path, output_path, class_names, imgsz=640, skip_frames=2):
    print(f"Running inference on video: {video_path}")
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))

    print(f"Video properties: {frame_width}x{frame_height} @ {fps}fps")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps // (skip_frames + 1), (frame_width, frame_height))

    frame_queue = Queue(maxsize=1)
    result_queue = Queue(maxsize=1)

    def inference_thread():
        print("Inference thread started")
        while True:
            if frame_queue.empty():
                continue
            frame = frame_queue.get()
            if frame is None:
                print("Inference thread ending")
                break
            result = process_frame(session, frame, imgsz=imgsz)
            if not result_queue.full():
                result_queue.put(result)

    Thread(target=inference_thread, daemon=True).start()

    frame_count = 0
    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("End of video reached")
            break

        frame_count += 1
        if frame_count % (skip_frames + 1) != 0:
            continue

        if frame_queue.empty():
            frame_queue.put(frame)

        if not result_queue.empty():
            result = result_queue.get()
            frame = draw_boxes(frame, result, class_names)
            out.write(frame)

        cv2.imshow('Object Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("User interrupted")
            break

    frame_queue.put(None)
    video_capture.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Output video saved to {output_path}")

def run_inference_on_webcam(session, output_path, class_names, imgsz=640, skip_frames=2):
    print("Starting webcam inference")
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Error: Could not open webcam")
        return

    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))

    print(f"Webcam properties: {frame_width}x{frame_height} @ {fps}fps")

    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps // (skip_frames + 1), (frame_width, frame_height))

    frame_queue = Queue(maxsize=1)
    result_queue = Queue(maxsize=1)

    def inference_thread():
        print("Inference thread started")
        while True:
            if frame_queue.empty():
                continue
            frame = frame_queue.get()
            if frame is None:
                print("Inference thread ending")
                break
            result = process_frame(session, frame, imgsz=imgsz)
            if not result_queue.full():
                result_queue.put(result)

    Thread(target=inference_thread, daemon=True).start()

    frame_count = 0
    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Error reading from webcam")
            break

        frame_count += 1
        if frame_count % (skip_frames + 1) != 0:
            continue

        if frame_queue.empty():
            frame_queue.put(frame)

        if not result_queue.empty():
            result = result_queue.get()
            frame = draw_boxes(frame, result, class_names)

            if output_path:
                out.write(frame)

            cv2.imshow('Object Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("User interrupted")
            break

    frame_queue.put(None)
    video_capture.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()
    if output_path:
        print(f"Output video saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run object detection on image, video, or webcam')
    parser.add_argument('--weights', required=True, help='Path to the ONNX weights file')
    parser.add_argument('--input-type', choices=['image', 'video', 'webcam'], required=True, help='Input type (image, video, or webcam)')
    parser.add_argument('--input-path', help='Path to the input image or video file')
    parser.add_argument('--output-path', required=True, help='Path to save the output image or video')
    parser.add_argument('--imgsz', type=int, default=640, help='Input size for the model')
    parser.add_argument('--skip-frames', type=int, default=2, help='Number of frames to skip between detections')
    parser.add_argument('--class-names', required=True, help='Path to the class names file (JSON or TXT)')
    args = parser.parse_args()

    print(f"Using ONNX Runtime for inference")
    print(f"ONNX Runtime version: {ort.__version__}")
    print(f"OpenCV version: {cv2.__version__}")
    print(f"Python version: {sys.version}")

    try:
        session = load_model(args.weights)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    try:
        class_names = load_class_names(args.class_names)
    except Exception as e:
        print(f"Error loading class names: {e}")
        sys.exit(1)

    print(f"Input type: {args.input_type}")
    print(f"Input path: {args.input_path}")
    print(f"Output path: {args.output_path}")
    print(f"Image size: {args.imgsz}")
    print(f"Skip frames: {args.skip_frames}")

    if args.input_type == 'image':
        run_inference_on_image(session, args.input_path, args.output_path, class_names, imgsz=args.imgsz)
    elif args.input_type == 'video':
        run_inference_on_video(session, args.input_path, args.output_path, class_names, imgsz=args.imgsz, skip_frames=args.skip_frames)
    elif args.input_type == 'webcam':
        run_inference_on_webcam(session, args.output_path, class_names, imgsz=args.imgsz, skip_frames=args.skip_frames)
