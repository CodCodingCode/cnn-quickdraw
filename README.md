# Image Classification with TensorFlow
This project demonstrates a simple image classification application using TensorFlow and Keras. It includes a script that loads image data stored in .npy format, preprocesses it, and trains a Convolutional Neural Network (CNN) for classification. The model predicts the class of input images among a fixed set of classes derived from the Quick, Draw! dataset.

## Features
Data loading from .npy files.
Data preprocessing including normalization and one-hot encoding.
A Convolutional Neural Network implementation with TensorFlow's Keras API.
Training and evaluation of the model on the dataset.
Displaying top-5 predictions for random test images.

## Data Preparation
This project uses image data from the Quick, Draw! dataset. The dataset consists of millions of drawings across hundreds of categories, contributed by players of the game "Quick, Draw!".

Your image data should be in .npy format, stored in a directory named data at the root of the project. Each .npy file should contain images of a particular class. The script expects the images to be flattened into vectors of size 784 (corresponding to 28x28 pixel images).

## How It Works
Data Loading: The script first loads the image data and labels from .npy files.
Preprocessing: Images are reshaped and normalized, and labels are one-hot encoded.
Model Training: A CNN is defined and trained on the preprocessed data.
Evaluation: The trained model is evaluated on a test set, and a test loss and accuracy are printed.
Prediction: The model makes predictions on randomly selected test images to demonstrate its performance.
## Customization
You can adjust the hyperparameters and model architecture in the model function to experiment with different configurations.

## License
This project is licensed under the MIT License.
