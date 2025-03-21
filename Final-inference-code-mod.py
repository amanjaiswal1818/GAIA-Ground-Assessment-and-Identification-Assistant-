import argparse
import cv2
import numpy as np
from ultralytics import YOLO
import base64
from email.mime.text import MIMEText
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import os.path
import pickle

# If modifying these scopes, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/gmail.send']

def get_gmail_service():
    creds = None
    # The file token.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first time.
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    service = build('gmail', 'v1', credentials=creds)
    return service

def send_email(sender_email, recipient_email, subject, body):
    try:
        service = get_gmail_service()
        message = MIMEText(body)
        message['to'] = recipient_email
        message['from'] = sender_email
        message['subject'] = subject
        raw_message = base64.urlsafe_b64encode(message.as_string().encode("utf-8"))
        message = service.users().messages().send(userId="me", body={'raw': raw_message.decode("utf-8")}).execute()
        print(f"Message Id: {message['id']}")
        print("Email sent successfully")
    except Exception as e:
        print(f"Error sending email: {e}")

def process_image(model, input_path, output_path, sender_email, recipient_email):
    # Load the image
    img = cv2.imread(input_path)
    
    # Run inference
    results = model(img)
    
    ph_detected = False
    
    # Draw bounding boxes
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
    
    # Save the output image
    cv2.imwrite(output_path, img)
    
    # Send email if PH detected
    if ph_detected:
        subject = "Pothole Detected"
        body = f"A pothole has been detected in the image: {input_path}"
        send_email(sender_email, recipient_email, subject, body)

def process_video(model, input_path, output_path, frame_skip, sender_email, recipient_email):
    # Load the video
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create a video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps / frame_skip, (width, height))
    
    frame_count = 0
    ph_detected = False
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process every nth frame
        if frame_count % frame_skip == 0:
            # Run inference
            results = model(frame)
            
            # Draw bounding boxes
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
            
            # Write the output frame
            out.write(frame)
        
        frame_count += 1
    
    cap.release()
    out.release()
    
    # Send email if PH detected
    if ph_detected:
        subject = "Pothole Detected!"
        body = f"A pothole has been detected in the video: {input_path}"
        send_email(sender_email, recipient_email, subject, body)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='YOLOv10 Inference on Jetson')
    parser.add_argument('--weights', type=str, help='Path to the model weights file (pt or onnx)')
    parser.add_argument('--input-type', type=str, choices=['img', 'video'], help='Type of input (image or video)')
    parser.add_argument('--input-path', type=str, help='Path to the input image or video')
    parser.add_argument('--output-path', type=str, help='Path to the output image or video')
    parser.add_argument('--frame-skip', type=int, default=5, help='Number of frames to skip between processing')
    parser.add_argument('--recipient-email', type=str, default='anshulgada05@gmail.com', help='Recipient email address')
    args = parser.parse_args()
    
    # Email configuration
    sender_email = "teamgaiaai@gmail.com"
    recipient_email = args.recipient_email
    
    # Load the model
    model = YOLO(args.weights)
    
    if args.input_type == 'img':
        process_image(model, args.input_path, args.output_path, sender_email, recipient_email)
    elif args.input_type == 'video':
        process_video(model, args.input_path, args.output_path, frame_skip=args.frame_skip, sender_email=sender_email, recipient_email=recipient_email)

# Example usage:
# python3 Final-inference-code-mod-2.py --weights best.pt --input-type video --input-path /home/gaia/Desktop/PDS/test-vid.mp4 --output-path /home/gaia/Desktop/PDS/WP-Res-New-2.mp4 --recipient-emails anshulgada02@gmail.com amanj001818@gmail.com 210303105259@paruluniversity.ac.in --attach-result