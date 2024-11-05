
# Text Toxicity Detector

This project is designed to detect toxic language in user-submitted text. The goal is to identify comments with toxic content using a neural network model trained on a labeled dataset. The project leverages TensorFlow for model training, Gradio for interactive predictions, and custom vocabulary processing.

# Project Overview

This project trains a model to classify text as toxic or non-toxic based on word content. It uses a TensorFlow neural network with text preprocessing via the TextVectorization layer, and provides an interactive Gradio interface for testing the model with user input.

# Features

- Model Training: Uses a custom vocabulary and TextVectorization for preprocessing.
- Toxicity Detection: Classifies text based on toxicity probability scores.
- Interactive Interface: Gradio is used to create an easy-to-use interface for real-time text classification.

# Installation

- Python 3.x
- Virtual Environment (recommended)
- Required libraries: TensorFlow, Gradio, Pandas, NumPy

# Model Training and Saving

" python train_model.py "

- Load and preprocess the data.
- Train the neural network model on the dataset.
- Save the trained model to toxicity.h5 file.

# Loading and Using the Model for Predictions

- Load the saved model and make predictions by running the predict.py script.

# Interactive Predictions with Gradio

- Interactive Predictions with Gradio after running predict.py

# Model Training and Evaluation

- Preprocess Data: The dataset is preprocessed using the TextVectorization layer in TensorFlow, with a custom vocabulary loaded from vocab.txt.

- Train the Model: The neural network is trained on the toxicity dataset, optimizing for accuracy and efficiency. A max_tokens limit is set on the vocabulary to avoid memory issues.

- Save the Model: The trained model is saved to toxicity.h5 for later use.

# Future Improvements

- Hyperparameter Tuning: Experiment with different model architectures and hyperparameters for better performance.
- Additional Features: Add support for more nuanced classifications (e.g., hate speech, offensive language, etc.).
- Advanced Deployment: Deploy the model on a cloud platform or integrate it with existing web applications.











