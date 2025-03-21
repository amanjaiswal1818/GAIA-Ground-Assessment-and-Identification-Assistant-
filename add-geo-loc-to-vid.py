import subprocess

def add_gps_to_video(video_path, lat, lon, output_path):
    # Format latitude and longitude for separate metadata
    lat_key = f"latitude={lat}"
    lon_key = f"longitude={lon}"
    
    command = [
        "ffmpeg",
        "-i", video_path,
        "-metadata", lat_key,
        "-metadata", lon_key,
        "-map_metadata", "0",
        "-codec", "copy",
        output_path
    ]
    
    try:
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result.stdout.decode())
        print(result.stderr.decode())
        print(f"Video saved with GPS data: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e.stderr.decode()}")

# Example usage
add_gps_to_video("test-vid.mp4", 40.7128, -74.0060, "test-vid-with-gps.mp4")
