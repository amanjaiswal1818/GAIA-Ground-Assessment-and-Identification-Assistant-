import re
import cv2
import json
import time
import base64
import pickle
import piexif
import os.path
import argparse
import subprocess
import numpy as np
from PIL import Image
from ultralytics import YOLO
from fractions import Fraction

from email.mime.text import MIMEText
from googleapiclient.discovery import build
from email.mime.multipart import MIMEMultipart
from google.oauth2.credentials import Credentials
from email.mime.application import MIMEApplication
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow


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
        gps_data = get_gps_from_image(input_path)
        
        subject = "Pothole Detected in Image!"
        body = f"A pothole has been detected in the image: {os.path.basename(input_path)}\n\n"
        body += f"Output image: {os.path.basename(output_path)}\n\n"
        body += f"Inference time: {inference_time:.2f} seconds\n\n"
        
        if gps_data:
            lat, lon = gps_data
            body += f"GPS Coordinates: \nLatitude: {lat}, \nLongitude: {lon}\n"
            body += f"Google Maps Link: https://www.google.com/maps?q={lat},{lon}\n"
        else:
            body += "No geolocation data exists in the input file.\n"
        
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
        gps_data = get_gps_from_video(input_path)
        
        subject = "Pothole Detected in Video!"
        body = f"A pothole has been detected in the video: {os.path.basename(input_path)}\n\n"
        body += f"Output video: {os.path.basename(output_path)}\n\n"
        body += f"Video duration: {video_duration:.2f} seconds\n\n"
        body += f"Inference time: {inference_time:.2f} seconds\n\n"
        
        if gps_data:
            lat, lon = gps_data
            body += f"GPS Coordinates: \nLatitude: {lat}, \nLongitude: {lon}\n"
            body += f"Google Maps Link: https://www.google.com/maps?q={lat},{lon}\n"
        else:
            body += "No geolocation data exists in the input file.\n"
        
        send_email(sender_email, recipient_emails, subject, body, input_path, output_path)
    
    return inference_time, video_duration
         
# OLD
# def get_gps_from_image(image_path):  
#     try:
#         img = Image.open(image_path)
#         exif_data = img._getexif()
#         if exif_data:
#             exif_dict = piexif.load(img.info["exif"])
#             if piexif.GPSIFD.GPSLatitude in exif_dict["GPS"]:
#                 lat = exif_dict["GPS"][piexif.GPSIFD.GPSLatitude]
#                 lon = exif_dict["GPS"][piexif.GPSIFD.GPSLongitude]
#                 lat_ref = exif_dict["GPS"][piexif.GPSIFD.GPSLatitudeRef]
#                 lon_ref = exif_dict["GPS"][piexif.GPSIFD.GPSLongitudeRef]
                
#                 lat = sum(float(num)/float(denom) for num, denom in lat) * (-1 if lat_ref == "S" else 1)
#                 lon = sum(float(num)/float(denom) for num, denom in lon) * (-1 if lon_ref == "W" else 1)
                
#                 return lat, lon
#     except Exception as e:
#         print(f"Error extracting GPS data from image: {e}")
#     return None

def get_gps_from_image(image_path):
    try:
        img = Image.open(image_path)
        exif_data = img._getexif()
        if exif_data:
            exif_dict = piexif.load(img.info["exif"])
            if piexif.GPSIFD.GPSLatitude in exif_dict["GPS"]:
                lat = exif_dict["GPS"][piexif.GPSIFD.GPSLatitude]
                lon = exif_dict["GPS"][piexif.GPSIFD.GPSLongitude]
                lat_ref = exif_dict["GPS"][piexif.GPSIFD.GPSLatitudeRef].decode()
                lon_ref = exif_dict["GPS"][piexif.GPSIFD.GPSLongitudeRef].decode() 
                
                def convert_to_degrees(value):
                    d = float(value[0][0]) / float(value[0][1])
                    m = float(value[1][0]) / float(value[1][1])
                    s = float(value[2][0]) / float(value[2][1])
                    return d + (m / 60.0) + (s / 3600.0)
                
                lat = convert_to_degrees(lat)
                lon = convert_to_degrees(lon)
                
                if lat_ref == "S":
                    lat = -lat
                if lon_ref == "W":
                    lon = -lon
                
                return lat, lon
    except Exception as e:
        print(f"Error extracting GPS data from image: {e}")
    return None


def get_gps_from_video(video_path):
    try:
        # Use ffprobe to get video metadata
        cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', video_path]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        metadata = json.loads(result.stdout)
        
        # Check for GPS data in various metadata fields
        gps_data = None
        if 'format' in metadata and 'tags' in metadata['format']:
            tags = metadata['format']['tags']
            if 'location' in tags:
                gps_data = tags['location']
            elif 'com.apple.quicktime.location.ISO6709' in tags:
                gps_data = tags['com.apple.quicktime.location.ISO6709']
        
        if gps_data:
            # Parse GPS data
            match = re.search(r'([+-]\d+\.\d+)([+-]\d+\.\d+)', gps_data)
            if match:
                lat, lon = map(float, match.groups())
                return lat, lon
            else:
                # Try parsing degrees, minutes, seconds format
                match = re.search(r'([+-])(\d+)(\d{2})(\d{2})([+-])(\d+)(\d{2})(\d{2})', gps_data)
                if match:
                    lat_sign, lat_d, lat_m, lat_s, lon_sign, lon_d, lon_m, lon_s = match.groups()
                    lat = int(lat_d) + int(lat_m)/60 + int(lat_s)/3600
                    lon = int(lon_d) + int(lon_m)/60 + int(lon_s)/3600
                    if lat_sign == '-':
                        lat = -lat
                    if lon_sign == '-':
                        lon = -lon
                    return lat, lon
        
        print("No GPS data found in video metadata")
        return None
    except Exception as e:
        print(f"Error extracting GPS data from video: {e}")
    return None


# OLD
# def get_gps_from_video(video_path):
#     try:
#         # Use ffmpeg to get video metadata
#         cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format_tags=location', '-of', 'json', video_path]
#         result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)        
#         metadata = json.loads(result.stdout)
#         if 'format' in metadata and 'tags' in metadata['format']:
#             tags = metadata['format']['tags']
#             if 'location' in tags:
#                 lat_lon = tags['location'].split(',')
#                 if len(lat_lon) == 2:
#                     return float(lat_lon[0]), float(lat_lon[1])   
#     except Exception as e:
#         print(f"Error extracting GPS data from video: {e}")
#     return None     
#     
#    
#     

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='YOLOv10 Inference on Jetson')
    parser.add_argument('--weights', type=str, help='Path to the model weights file (pt or onnx)')
    parser.add_argument('--input-type', type=str, choices=['img', 'video'], help='Type of input (image or video)').
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
# python3 Final-inference-code-mod-2.py --weights best.pt --input-type video --input-path /home/gaia/Desktop/PDS/test-vid.mp4 --output-path /home/gaia/Desktop/PDS/WP-Res-New-3.mp4 --recipient-emails amanj001818@gmail.com ankita.gandhi@paruluiniversity.ac.in kishori.shekokar20174@paruluniversity.ac.in --attach-result
# python3 Final-inference-code-mod-3.py --weights best.pt --input-type video --input-path /home/gaia/Desktop/PDS/test-vid-with-gps.mp4 --output-path /home/gaia/Desktop/PDS/test-vid-with-gps-res.mp4 --recipient-emails amanj001818@gmail.com anshulgada05@gmail.com --attach-result
# python3 Final-inference-code-mod-3.py --weights best.pt --input-type video --input-path /home/gaia/Desktop/PDS/test-vid.mp4 --output-path /home/gaia/Desktop/PDS/test-vid-res.mp4 --recipient-emails amanj001818@gmail.com anshulgada05@gmail.com --attach-result