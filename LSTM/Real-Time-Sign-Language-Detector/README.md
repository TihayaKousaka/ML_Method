# Real-Time Sign Language Detection using LSTM in TensorFlow

## Project Overview:
This GitHub repository contains a real-time sign language detection project built using deep learning techniques with TensorFlow. The project aims to recognize sign language gestures in real-time using an LSTM-based deep learning model. The system tracks the face, hands, and body keypoints using Mediapipe's holistic models and leverages this sequential information to make accurate predictions.

## Project Structure:
data_generator.ipynb: This Jupyter notebook is responsible for generating the training data. It uses OpenCV to record images from a camera and utilizes Mediapipe's holistic models to trace and record the face, hands, and body keypoints.

model_trainer.ipynb: In this Jupyter notebook, the LSTM-based deep learning model is designed and trained on the dataset generated by the data generator. LSTM is chosen due to its ability to capture sequential information, which is crucial for detecting the sequence of sign language actions.

realtime_tester.ipynb: The realtime_tester notebook uses the trained model to perform real-time sign language detection. It captures video input from the camera using OpenCV and then utilizes the LSTM model to predict sign language gestures in real-time.

## Getting Started:

1. Clone the repository to your local machine:

2. Set-up the environment in your machine. Here is the list of dependencies used:
   * Python : 3.11.4
   * Tensorflow: 2.12
   * OpenCV: 4.8
   * Mediapipe: 0.10.2
   * Numpy: 1.23.5
   * Matplotlib: 3.7.2

3. Data Generation: Execute the data_generator.ipynb notebook to generate training data by recording images and extracting keypoints. The data is then stored in a directory called MP_Data. 

4. Model Training: Run the model_trainer.ipynb notebook to design and train the LSTM-based deep learning model using the generated dataset, which is imported from the MP_Data directory.

5. Real-Time Detection: Launch the realtime_tester.ipynb notebook to perform real-time sign language detection using your camera feed.
