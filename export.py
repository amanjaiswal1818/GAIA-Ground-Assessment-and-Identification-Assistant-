from ultralytics import YOLO

try:
    print('\nProcessing...')
    
    # Load the YOLOv10 model
    model = YOLO("best.pt")

    print('\nModel Loaded...')
    
    # Export the model to ONNX format
    model.export(format="onnx")  # creates 'yolov10m.onnx'
    
    print('\nConversion Done')

except:
    print("error")