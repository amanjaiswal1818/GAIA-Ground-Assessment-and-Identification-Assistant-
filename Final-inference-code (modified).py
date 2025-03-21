import argparse
import cv2
import numpy as np
from ultralytics import YOLO
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_email(sender_email, sender_password, recipient_email, subject, body):
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = recipient_email
    message["Subject"] = subject
    message.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(message)
        print("Email sent successfully")
    except Exception as e:
        print(f"Error sending email: {e}")

def process_image(model, input_path, output_path, sender_email, sender_password, recipient_email):
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
        send_email(sender_email, sender_password, recipient_email, subject, body)

def process_video(model, input_path, output_path, frame_skip, sender_email, sender_password, recipient_email):
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
        subject = "Pothole Detected"
        body = f"A pothole has been detected in the video: {input_path}"
        send_email(sender_email, sender_password, recipient_email, subject, body)

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
    sender_password = "G@ia2024"
    recipient_email = args.recipient_email
    
    # Load the model
    model = YOLO(args.weights)
    
    if args.input_type == 'img':
        process_image(model, args.input_path, args.output_path, sender_email, sender_password, recipient_email)
    elif args.input_type == 'video':
        process_video(model, args.input_path, args.output_path, frame_skip=args.frame_skip, sender_email=sender_email, sender_password=sender_password, recipient_email=recipient_email)

# Example usage:
# python3 yolo-inference-script.py --weights best.pt --input-type video --input-path /home/gaia/Desktop/PDS/videoplayback.mp4 --output-path /home/gaia/Desktop/PDS/videoplayback-res-low-fps.mp4 --frame-skip 5 --recipient-email custom@example.com
