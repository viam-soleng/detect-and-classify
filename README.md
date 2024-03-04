# Viam Cropping Vision Service

This repository contains the `visionsvc` package, a module of the Viam vision service designed for image cropping and further analysis. It integrates several vision services, including object detection, age classification, and gender classification.

## Description

The Viam Cropping Vision Service (`visionsvc`) is a specialized module within the Viam vision framework. Its primary function is to crop an image to an initial detection and then utilize other models to return detailed detections, including age and gender classifications.

![alt text](media/architecture.png "Cropping Service Architecture")

## Features

- Takes a camera as input
- Uses an object detector to identify the objects bounding boxes
- Crops the detected images according to their bounding boxes
- Feeds the cropped images into the configured classifier for more accurate results
- Returns the classification labels and score

## Configuration and Dependencies

Dependencies are implicit.

Sample Configuration Attributes:
```json
{
  "camera": "camera",
  "detector_service": "detector",
  "detector_confidence": 0.5,
  "detector_detections": 5,
  "detector_valid_labels": [
    "label"
  ],
  "classifier_service": "classifier",
  "classifier_results": 1,
  "log_image": false,             //Optional
  "image_path": "<- YOUR PATH ->" //Optional
}
```

Sample Component Configuration:
```json
    {
      "model": "sol-eng:vision:cropping-service",
      "attributes": {
        "classifier_service": "vision-age",
        "detector_service": "vision-face",
        "camera": "cam",
        "detector_confidence": 0.7,
        "detector_detections": 5,
        "log_image": false,
        "image_path": "/Users/Username/faces/",
        "detector_valid_labels": [
          "face"
        ],
        "classifier_results": 1
      },
      "name": "cropping",
      "type": "vision",
      "namespace": "rdk"
    }
```
