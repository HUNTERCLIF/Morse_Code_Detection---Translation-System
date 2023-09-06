# Morse_Code_Detection---Translation-System

Introduction

The Morse Code Detection and Translation System is a project designed to detect and translate Morse code signals from audio recordings into readable text. This README provides an overview of the project's features, how to use it, and the setup process.
Features

    Audio files are converted into spectrogram images for analysis.
    A convolutional neural network (CNN) is trained to detect Morse code symbols.
    Detected symbols are translated into readable text.
    Provides boundary detection for individual letters in Morse code.
    Evaluates accuracy, precision, recall, and F1-score for Morse code detection.

Usage

This section describes how to use the Morse Code Detection and Translation System:

    Setup: Follow the setup instructions to prepare your environment.

    Training: Train the CNN model to detect Morse code symbols using labeled data.

    Testing: Test the trained model on audio files containing Morse code signals.

    Letter Detection: Detect boundaries of individual letters in Morse code signals.

    Morse Code Translation: Translate detected Morse code symbols into readable text.

Setup

Before using the system, you need to set up your environment:

    Install the required libraries:
        Organize your Morse code dataset as per the project structure.

    Create the necessary folders for training, validation, and testing.

Training

To train the CNN model:

    Ensure you have organized the Morse code dataset into train, validation, and test folders.
    Run the training script:

    Testing

To test the Morse code detection model:

    Prepare your test data in the appropriate format.

    Run the testing script:
    The system will provide accuracy, precision, recall, and F1-score metrics.

Letter Detection

The system can also detect boundaries of individual letters in Morse code signals. Refer to the Letter Detection section in the code for details.
Morse Code Translation

Detected Morse code symbols can be translated into readable text using the provided Morse code dictionary. Refer to the Morse Code Translation section in the code for details.
    
    
