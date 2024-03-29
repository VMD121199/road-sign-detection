# Road Sign Detection Project

## Overview

This project aims to develop a robust and accurate road sign detection system using PyTorch, Tensorflow, and Yolov8. The system is designed to identify and locate various road signs in images and video streams, contributing to enhanced road safety and autonomous driving capabilities.

## Features

- **Road Sign Detection:** Utilize state-of-the-art algorithms to detect and classify different road signs.
- **Real-time Processing:** Achieve real-time performance for road sign detection in live video streams.
- **Customization:** Easily extend the system to recognize additional road sign categories or adapt to specific use cases.

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/VMD121199/road-sign-detection.git
   cd road-sign-detection
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

<!-- 3. [Add any specific installation instructions or additional dependencies if needed.] -->

## Model Training

My road sign detection model was trained using a diverse dataset of road sign images collected from various sources. The dataset comprises 4 categories of road signs, totaling 877 images. I utilized the SSD300 implemented with VGG16 from Keras for training, I use one more method Yolov8 to compare the performance of 2 methods SSD and YOLO.

### Training Process

1. **Data Preprocessing:** Images were resized to a standard input size of (300, 300) for SSD and (320, 320) for Yolov8.

2. **Model Configuration:** I configured the VGG16 with a carefully chosen learning rate, optimizer, loss function, batch size, and number of epochs for optimal performance.

3. **Training:** The model was trained for 15 epochs using the Adam optimizer, cost is calculated by location loss and label loss.

4. **Evaluation:** The trained model was evaluated on a separate validation dataset, achieving an accuracy of 78.71% for label accuracy and 70.68% for bounding box accuracy.

5. **Fine-tuning (Optional):** Fine-tuning on specific road sign categories or regions of interest was performed to improve model accuracy in critical areas.

## Results

Our road sign detection system demonstrates robust performance across various scenarios. Below are some visual examples showcasing the system's capabilities:


**SSD300:**

![Example 2](images/image_2.png)

**YOLO v8:**

![Example 1](images/image_1.png)

### Performance Metrics

1. **SSD:**

- **Label Accuracy:** 78.71%
- **Bounding box accuracy:** 70.68%

2. **Yolov8:**

- **Precision:** 86.9%
- **Recall:** 81.2%
- **mAP50:** 87.1%
- **mAP50-95:** 71.7%

## Contributing

I always welcome contributions from the community to enhance and improve this road sign detection project. If you would like to contribute, please follow these guidelines:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and submit a pull request.

Please ensure that your code adheres to our coding standards and includes relevant tests.

<!-- ## Acknowledgments

We would like to express our gratitude to the following:

- [Name of external library or framework] for providing essential functionality.
- [Name of dataset provider] for the road sign dataset used in training. -->
