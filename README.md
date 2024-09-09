# demo-face-blur
Video anonymization software in Python that blurs people's faces using OpenCV, YuNet.
[Check out my post on blurring faces @ medium.com](https://easyeasy.medium.com/protecting-privacy-a-comprehensive-guide-to-video-anonymization-for-ai-training-4b85fb23a61d)

## Requirements
- Python 3.9
- OpenCV 4.7.0

OpenCV is not part of requirements.txt as I am using a custom build with CUDA support. 
You should be able to `pip install opencv-python==4.7.0` 

## Usage
1. add model files to `models` folder. Download:
[face_detection_yunet_2022mar.onnx](https://github.com/opencv/opencv_zoo/blob/7e062e54cf5410c09b795ff71b4a255e58498c79/models/face_detection_yunet/face_detection_yunet_2022mar.onnx).
2. `pip install -r requirements.txt`
3. install OpenCV (see Requirements)
4. run `python main.py input.mp4`

## Credits
Multi-threading video display code Copyright (c) 2018 Najam Syed (MIT License)
https://nrsyed.com/2018/07/05/multithreading-with-opencv-python-to-improve-video-processing-performance/
https://github.com/nrsyed/computer-vision/tree/master/multithread
