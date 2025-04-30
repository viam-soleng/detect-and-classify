# Viam Detect and Classify Vision Service

This repository contains the `visionsvc` package, a module of the Viam vision service designed for optimizing input images for classifier and detction vision services. It can use a camera or an image as input. This image is then procesed with the object detector service to get the bounding boxes. Depending on the configured vision service, you can then get the result through either the classifications or detections API of this detect-and-classify vision service. Cropping images before processing them with further more specialiced ml models dramatically improves result accuracy.

Available via the [Viam Registry](https://app.viam.com/module/viam-soleng/detect-and-classify)! -> Currently for darwin/arm64 and android/arm64 others will follow soon.

## Description

The Viam Detect and Classify Vision Service (`visionsvc`) is a specialized module within the Viam vision framework. Its primary function is to crop an image to an initial detection and then utilize a classifier or detector model to return accurate classifications/detections. An example could be to use a face detector and then an age classifier model.

![alt text](media/architecture.png "Detect and Classify Service Architecture")

## Features

- Takes a camera as input
- Uses an object detector to identify the objects bounding boxes
- Crops the detected images according to their bounding boxes
- Feeds the cropped images into the configured classifier/detector
- Returns the classifications/detections

## Configuration and Dependencies

Dependencies are implicit.

Sample Configuration Attributes:

```json
{
  "camera": "camera",
  "detector_service": "object-detector",
  "detector_confidence": 0.5,
  "detector_valid_labels": ["label"],
  "max_detections": 5,
  "padding": 30,
  "vision_service": "classifier or detector vision service",
  "log_images": false,
  "images_path": "<- YOUR PATH ->"
}
```
