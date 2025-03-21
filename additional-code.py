from PIL import Image
import piexif
from fractions import Fraction
import cv2

def get_gps_from_image(image_path):
    try:
        img = Image.open(image_path)
        exif_data = img._getexif()
        if exif_data:
            exif_dict = piexif.load(img.info["exif"])
            if piexif.GPSIFD.GPSLatitude in exif_dict["GPS"]:
                lat = exif_dict["GPS"][piexif.GPSIFD.GPSLatitude]
                lon = exif_dict["GPS"][piexif.GPSIFD.GPSLongitude]
                lat_ref = exif_dict["GPS"][piexif.GPSIFD.GPSLatitudeRef]
                lon_ref = exif_dict["GPS"][piexif.GPSIFD.GPSLongitudeRef]
                
                lat = sum(float(num)/float(denom) for num, denom in lat) * (-1 if lat_ref == "S" else 1)
                lon = sum(float(num)/float(denom) for num, denom in lon) * (-1 if lon_ref == "W" else 1)
                
                return lat, lon
    except Exception as e:
        print(f"Error extracting GPS data from image: {e}")
    return None

def get_gps_from_video(video_path):
    try:
        video = cv2.VideoCapture(video_path)
        if video.isOpened():
            metadata = video.get(cv2.CAP_PROP_METADATA)
            if metadata:
                # This is a simplification. Actual parsing might be more complex
                lat, lon = metadata.split('/')
                return float(lat), float(lon)
    except Exception as e:
        print(f"Error extracting GPS data from video: {e}")
    return None

def process_image(model, input_path, output_path, sender_email, recipient_emails):
    # ... (existing code) ...
    
    if ph_detected:
        gps_data = get_gps_from_image(input_path)
        
        subject = "Pothole Detected in Image"
        body = f"A pothole has been detected in the image: {os.path.basename(input_path)}\n"
        body += f"Output image: {os.path.basename(output_path)}\n"
        body += f"Inference time: {inference_time:.2f} seconds\n"
        
        if gps_data:
            lat, lon = gps_data
            body += f"GPS Coordinates: Latitude {lat}, Longitude {lon}\n"
            body += f"Google Maps Link: https://www.google.com/maps?q={lat},{lon}\n"
        else:
            body += "No geolocation data exists in the input file.\n"
        
        send_email(sender_email, recipient_emails, subject, body, input_path, output_path)

def process_video(model, input_path, output_path, frame_skip, sender_email, recipient_emails):
    # ... (existing code) ...
    
    if ph_detected:
        gps_data = get_gps_from_video(input_path)
        
        subject = "Pothole Detected in Video"
        body = f"A pothole has been detected in the video: {os.path.basename(input_path)}\n"
        body += f"Output video: {os.path.basename(output_path)}\n"
        body += f"Video duration: {video_duration:.2f} seconds\n"
        body += f"Inference time: {inference_time:.2f} seconds\n"
        
        if gps_data:
            lat, lon = gps_data
            body += f"GPS Coordinates: Latitude {lat}, Longitude {lon}\n"
            body += f"Google Maps Link: https://www.google.com/maps?q={lat},{lon}\n"
        else:
            body += "No geolocation data exists in the input file.\n"
        
        send_email(sender_email, recipient_emails, subject, body, input_path, output_path)


# pip install Pillow piexif