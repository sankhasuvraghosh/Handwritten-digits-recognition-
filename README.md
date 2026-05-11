Handwritten Digit Recognition using Neural Networks :

This project implements a handwritten digit recognition system using a deep neural network built with PyTorch. The model is trained on the MNIST dataset, which contains grayscale images of handwritten digits (0–9), and learns to classify them with high accuracy.

The neural network consists of multiple fully connected layers with ReLU activation functions, optimized using the Adam optimizer and trained with CrossEntropy Loss. The project demonstrates the complete deep learning workflow including data preprocessing, model building, training, evaluation, saving trained weights, and predicting custom digit samples.

Features
Data preprocessing using torchvision transforms
Neural network implementation with PyTorch
GPU acceleration support (CUDA)
Training and testing pipeline
Accuracy evaluation on unseen test data
Model saving and loading
Single image digit prediction
Visualization of handwritten digit samples
Technologies Used
Python
PyTorch
Torchvision
Matplotlib
MNIST Dataset
Model Architecture

Input Layer:

784 neurons (28×28 flattened image)

Hidden Layers:

Dense Layer (128 neurons, ReLU)
Dense Layer (64 neurons, ReLU)

Output Layer:

Dense Layer (10 neurons for digit classes 0–9)
Training Details
Optimizer: Adam
Loss Function: CrossEntropyLoss
Batch Size: 64
Epochs: 10
Device Support: CPU / GPU
Workflow
Load and preprocess MNIST dataset
Build neural network model
Train using forward and backward propagation
Optimize weights using Adam optimizer
Evaluate model accuracy
Save trained model
Predict handwritten digits from test samples
Applications

This project demonstrates practical applications of neural networks in:

Optical Character Recognition (OCR)
Automated form digitization
Bank cheque digit reading
Postal code recognition
Document processing systems
Results

The trained model achieves high classification accuracy on the MNIST test dataset and successfully predicts handwritten digits from unseen samples.
