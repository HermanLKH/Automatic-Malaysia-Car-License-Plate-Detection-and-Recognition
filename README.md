# Automatic Malaysia Car License Plate Detection and Recognition

![Application Screenshot](assets/application_screenshot.png)

---

This repository implements a complete ALPR pipeline optimized for Malaysian license plates. Our **best-performing pipeline** uses:

- **YOLOv8n** for detection (vehicle + license plate)  
- **CRNN-Attention** for recognition (optical character recognition)

Click into each subdirectory for step-by-step instructions on training and testing:

- `YOLO` – YOLOv8n setup, data preparation, training & evaluation  
- `CRNN` – CRNN-Attn data pipeline, training notebooks, evaluation  

## Other Models

- **Detection** – `Faster-RCNN`  
- **Recognition** – `PaddleOCR`  
- **Recognition** – `ABINET`  

## Resources

- **Live App**: [Try the Hugging Face Space](https://huggingface.co/spaces/hermanlkh/MalaysiaALPR)
- **Report**: [View the full report here](https://drive.google.com/file/d/1MboxH52VK3PkceWKifzyutCZuayyjl0C/view?usp=sharing)
- **Video Presentation**: [Watch the system in action](https://drive.google.com/file/d/1DdmPxM4dUoIaVN9FjRlzXrhciUg29RTb/view?usp=sharing)  

---
