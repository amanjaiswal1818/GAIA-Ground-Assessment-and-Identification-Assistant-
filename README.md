# GAIA: Ground Assessment and Identification Assistant

## Project Description

GAIA is an innovative pothole detection system designed to enhance road safety and infrastructure maintenance. By leveraging advanced technologies like YOLOv10, drones, and the NVIDIA Jetson Orin Nano, GAIA automates the detection of potholes with improved speed, accuracy, and resource allocation. 

## Key Features

* **Automated Pothole Detection:** Utilizes YOLOv10 for real-time object detection with high accuracy. 
    
* **Real-time Monitoring:** Employs drones and vehicle-mounted cameras for continuous road condition monitoring. 
    
* **Depth Estimation:** Measures pothole depth using stereo vision or LiDAR sensors. 
    
* **GPS Integration:** Accurate geolocation of detected potholes for efficient reporting and repair.
    
* **Data Transmission & Visualization:** Pothole data is transmitted to a central server for visualization and alerting of maintenance teams. 
    
* **Hardware Acceleration:** NVIDIA Jetson Orin Nano is used for efficient GPU-accelerated computing. 
    
* **Cost Efficiency:** Reduces the need for manual inspections, lowering operational costs. 
    
* **Improved Road Safety:** Enables timely maintenance, minimizing the risk of accidents and vehicle damage. 

## System Architecture

The system operates through the following stages:

1.  **Image Acquisition:** High-resolution cameras capture road images. 
   
2.  **Preprocessing:** Captured frames undergo noise reduction, contrast enhancement, and normalization. 
   
3.  **Pothole Detection:** The YOLOv10 model identifies potential potholes. 
   
4.  **Depth Estimation:** Stereo imaging or LiDAR measures pothole depth. 
   
5.  **Data Transmission:** Pothole depth and GPS coordinates are sent to a central server. 
   
6.  **Data Visualization and Alerting:** The central server updates the road condition map and alerts maintenance teams. 

## Implementation

The system utilizes:

* **YOLOv10:** A deep learning model for real-time object detection. 
   
* **NVIDIA Jetson Orin Nano:** Efficient GPU-accelerated computing for real-time inference. 
   
* **Drones and Cameras:** For image acquisition and road condition monitoring. 
   
* **LiDAR or Stereo Vision:** For depth estimation. 
   
* **GPS:** For accurate geolocation of potholes. 
   
* **Cloud Computing:** For data processing, storage, and visualization. 


## Image Test
![preview img](PH-Test-GT2.jpeg)
## Image Test-Res
![preview img](PH-Test-GT2-Res.jpeg)


## For Detailed Information Refer to the below Documents 
* GAIA Final Report.pdf
* GAIA PPT.pdf
* SRS pothole detection.pdf
* RP Final.pdf


## Team

* Anshul Gada
* Aman Jaiswal
* Juiee Yadav
* Ankita Gandhi
