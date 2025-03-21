import argparse
import cv2
import numpy as np
from ultralytics import YOLO
import base64
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import os.path
import pickle
import time

# If modifying these scopes, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/gmail.send']

def get_gmail_service():
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    service = build('gmail', 'v1', credentials=creds)
    return service

def send_email(sender_email, recipient_emails, subject, body, input_path, output_path):
    try:
        service = get_gmail_service()
        message = MIMEMultipart()
        message['to'] = ', '.join(recipient_emails)
        message['from'] = sender_email
        message['subject'] = subject
        
        message.attach(MIMEText(body))
        
        # Attach input file
        with open(input_path, "rb") as attachment:
            part = MIMEApplication(attachment.read(), Name=os.path.basename(input_path))
        part['Content-Disposition'] = f'attachment; filename="{os.path.basename(input_path)}"'
        message.attach(part)
        
        # Attach output file
        with open(output_path, "rb") as attachment:
            part = MIMEApplication(attachment.read(), Name=os.path.basename(output_path))
        part['Content-Disposition'] = f'attachment; filename="{os.path.basename(output_path)}"'
        message.attach(part)
        
        raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")
        message = service.users().messages().send(userId="me", body={'raw': raw_message}).execute()
        print(f"Message Id: {message['id']}")
        print("Email sent successfully")
    except Exception as e:
        print(f"Error sending email: {e}")

def process_image(model, input_path, output_path, sender_email, recipient_emails, attach_result):
    start_time = time.time()
    
    img = cv2.imread(input_path)
    results = model(img)
    
    ph_detected = False
    
    for result in results:
        if result.boxes is not None:
            for box in result.boxes.xyxy:
                x, y, x2, y2 = box
                class_id = result.boxes.cls[0]
                confidence = result.boxes.conf[0]
                class_name = model.names[int(class_id)]
                label = "{} {:.2f}".format(class_name, confidence)
                cv2.rectangle(img, (int(x), int(y)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(img, label, (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                if class_name.lower() == "ph" or class_name.lower() == "pothole":
                    ph_detected = True
    
    cv2.imwrite(output_path, img)
    
    end_time = time.time()
    inference_time = end_time - start_time
    
    if ph_detected:
        subject = "Pothole Detected in Image"
        body = f"A pothole has been detected in the image: {os.path.basename(input_path)}\n"
        body += f"Output image: {os.path.basename(output_path)}\n"
        body += f"Inference time: {inference_time:.2f} seconds"
        
        send_email(sender_email, recipient_emails, subject, body, input_path, output_path)
    
    return inference_time

def process_video(model, input_path, output_path, frame_skip, sender_email, recipient_emails, attach_result):
    start_time = time.time()
    
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps / frame_skip, (width, height))
    
    frame_count = 0
    ph_detected = False
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_skip == 0:
            results = model(frame)
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes.xyxy:
                        x, y, x2, y2 = box
                        class_id = result.boxes.cls[0]
                        confidence = result.boxes.conf[0]
                        class_name = model.names[int(class_id)]
                        label = "{} {:.2f}".format(class_name, confidence)
                        cv2.rectangle(frame, (int(x), int(y)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(frame, label, (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        
                        if class_name.lower() == "ph" or class_name.lower() == "pothole":
                            ph_detected = True
            
            out.write(frame)
        
        frame_count += 1
    
    cap.release()
    out.release()
    
    end_time = time.time()
    inference_time = end_time - start_time
    
    if ph_detected:
        subject = "Pothole Detected in Video"
        body = f"A pothole has been detected in the video: {os.path.basename(input_path)}\n\n"
        body += f"Output video: {os.path.basename(output_path)}\n\n"
        body += f"Video duration: {video_duration:.2f} seconds\n\n"
        body += f"Inference time: {inference_time:.2f} seconds\n"

        send_email(sender_email, recipient_emails, subject, body, input_path, output_path)
    
    return inference_time, video_duration

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='YOLOv10 Inference on Jetson')
    parser.add_argument('--weights', type=str, help='Path to the model weights file (pt or onnx)')
    parser.add_argument('--input-type', type=str, choices=['img', 'video'], help='Type of input (image or video)')
    parser.add_argument('--input-path', type=str, help='Path to the input image or video')
    parser.add_argument('--output-path', type=str, help='Path to the output image or video')
    parser.add_argument('--frame-skip', type=int, default=5, help='Number of frames to skip between processing')
    parser.add_argument('--recipient-emails', type=str, nargs='+', default=['anshulgada05@gmail.com'], help='Recipient email addresses')
    parser.add_argument('--attach-result', action='store_true', help='Attach the resulting image/video to the email')
    args = parser.parse_args()
    
    sender_email = "teamgaiaai@gmail.com"
    recipient_emails = args.recipient_emails
    
    model = YOLO(args.weights)
    
    if args.input_type == 'img':
        inference_time = process_image(model, args.input_path, args.output_path, sender_email, recipient_emails, args.attach_result)
        print(f"Inference time: {inference_time:.2f} seconds")
    elif args.input_type == 'video':
        inference_time, video_duration = process_video(model, args.input_path, args.output_path, frame_skip=args.frame_skip, sender_email=sender_email, recipient_emails=recipient_emails, attach_result=args.attach_result)
        print(f"Video duration: {video_duration:.2f} seconds")
        print(f"Inference time: {inference_time:.2f} seconds")

# Example usage:
# python3 yolo-inference-script.py --weights best.pt --input-type video --input-path /home/gaia/Desktop/PDS/videoplayback.mp4 --output-path /home/gaia/Desktop/PDS/videoplayback-res-low-fps.mp4 --frame-skip 5 --recipient-emails email1@example.com email2@example.com --attach-result
    