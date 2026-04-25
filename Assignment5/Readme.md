# Assignment 5 - Transfer Learning for Image and Text Classification

This assignment applies transfer learning using deep learning models for:
1. Image classification with Convolutional Neural Networks (CNNs)
2. Text classification with Transformer models

## 1. Image Classification (CNN)

- Dataset:
  - ~100 images of chickens
  - ~100 images of ducks
- Task:
  - Binary classification: chicken vs duck
- Method:
  - Fine-tune a pre-trained CNN (ResNet) in PyTorch
  - Replace the final layer for binary classification
- Output:
  - Classification report (accuracy, precision, recall, F1-score)

## 2. Text Classification (Transformer)

- Dataset:
  - [Kaggle Sentiment Analysis Dataset](https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset)
- Task:
  - Multi-class sentiment classification: positive, neutral, negative
- Method:
  - Fine-tune a pre-trained Transformer model (BERT)
  - Tokenize text and train using Hugging Face Transformers
- Output:
  - Classification report

## Outcome
Demonstrates how transfer learning improves performance on small datasets for both image and text classification tasks.
