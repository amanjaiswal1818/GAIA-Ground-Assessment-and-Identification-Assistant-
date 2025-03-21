from PIL import Image
import piexif
from fractions import Fraction

def decimal_to_dms(decimal):
    degrees = int(decimal)
    minutes = int((decimal - degrees) * 60)
    seconds = ((decimal - degrees) * 60 - minutes) * 60
    return (degrees, 1), (minutes, 1), (int(seconds * 100), 100)

def add_gps_to_image(image_path, lat, lon, output_path):
    img = Image.open(image_path)
    exif_dict = piexif.load(img.info["exif"]) if "exif" in img.info else {"GPS": {}}
    
    exif_dict["GPS"][piexif.GPSIFD.GPSLatitudeRef] = "N" if lat >= 0 else "S"
    exif_dict["GPS"][piexif.GPSIFD.GPSLatitude] = decimal_to_dms(abs(lat))
    exif_dict["GPS"][piexif.GPSIFD.GPSLongitudeRef] = "E" if lon >= 0 else "W"
    exif_dict["GPS"][piexif.GPSIFD.GPSLongitude] = decimal_to_dms(abs(lon))
    
    exif_bytes = piexif.dump(exif_dict)
    img.save(output_path, "jpeg", exif=exif_bytes)
    print(f"Image saved with GPS data: {output_path}")

# Example usage
add_gps_to_image("input_image.jpg", 40.7128, -74.0060, "output_image_with_gps.jpg")