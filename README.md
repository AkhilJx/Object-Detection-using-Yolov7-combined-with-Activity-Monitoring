# Object-Detection-using-Yolov7-combined-with-Activity-Monitoring
A Combination of Human Activity recognition using LSTM with Object detection using Yolov7

This Python script integrates multiple computer vision and deep learning techniques to perform real-time human activity recognition using a combination of pose estimation and object detection. Here's a breakdown of the key functionalities:

1. Import Libraries: The script begins by importing necessary libraries, including OpenCV, NumPy, and the MediaPipe library for holistic pose estimation and Onnxruntime: an open source project that is designed to accelerate machine learning across a wide range of frameworks, operating systems, and hardware platforms..

2.Initialize YOLOv7 Object Detector: An instance of the YOLOv7 object detector is initialized with a pre-trained model for detecting objects in the frame.

3.Set Up MediaPipe Holistic Model: The script configures the MediaPipe holistic model for pose estimation, which includes body landmarks.

4.Mediapipe Detection Functions: Functions are defined for performing pose estimation using the MediaPipe holistic model and for drawing landmarks on the image.

5.Angle and Distance Calculation Functions: Functions are defined to calculate angles and distances between specific body landmarks, which are used for analyzing the pose.

6.Standardization Functions: Two functions are provided for standardizing angle and keypoint values.

7.Coordinate Extraction Function: A function extracts specific coordinates from pose landmarks to calculate angles and distances.

8.Keypoint Extraction Function: Another function extracts both pose landmarks and additional features, standardizes them, and concatenates them into a single array.

9.Define Parameters for Human Activity Recognition: Parameters such as the actions to be detected, number of video sequences, and sequence length are set.

10.Label Mapping: Actions are mapped to numerical labels.

11.Define LSTM Neural Network Model: A sequential LSTM neural network model is defined and loaded with pre-trained weights.

12.Probability Visualization Function: A function is defined to visualize the probabilities of detected activities.

13.Real-Time Testing: The script captures video frames from the camera, performs object detection, pose estimation, and activity recognition. It uses an LSTM model to predict human activities and visualizes the results in real-time. The recognized activities are displayed on the screen.

14.Output Video: The script also generates an output video ('Output.avi') with overlaid information about detected activities. ie the output of the program is saved locally so that we can show it 

15.User Interaction: The script provides a user-friendly interface with instructions to press 'q' to exit.

This comprehensive script combines the strengths of YOLOv7 for object detection, MediaPipe for pose estimation, and LSTM for human activity recognition. The results are presented in a visually appealing and informative manner, making it suitable for monitoring and analyzing human activities in real-time video streams.

A bespoke model has been developed for human activity recognition, with the corresponding source code included in this repository. The code implementation for custom activity recognition is also provided alongside a sample output screen and a video showcasing the results. I invite any constructive feedback, suggestions, or identification of potential issues with the code. Your insights and contributions are highly appreciated.


Sample Output:
![image](https://github.com/AkhilJx/Object-Detection-using-Yolov7-combined-with-Activity-Monitoring-using-LSTM/assets/78065413/c2a0c6ca-d928-4904-a4c2-b4d8ea14395f)
