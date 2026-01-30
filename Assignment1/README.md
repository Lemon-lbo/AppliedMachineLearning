
# AML Assignment 1 - SMS Spam Classification Prototype

This repository contains a prototype for SMS spam classification using classical machine learning models.

## Repository Structure

* `prepare.ipynb` – Data loading, preprocessing, and dataset splitting
* `train.ipynb` – Model training, hyperparameter tuning, and evaluation
* `train.csv`, `validation.csv`, `test.csv` – Prepared dataset splits

### `prepare.ipynb`

This notebook:

* Loads the raw SMS dataset from the provided file path
* Cleans and preprocesses text data
* Splits the dataset into training, validation, and test sets
* Saves the splits as `train.csv`, `validation.csv`, and `test.csv`

### `train.ipynb`

This notebook:

* Trains three models using `train.csv`:

  * **Logistic Regression** – Interpretable linear baseline
  * **Naive Bayes Classifier** – Effective for sparse, high-dimensional text data
  * **Linear SVM** – Maximizes class separation for strong performance
* Fine-tunes model hyperparameters:

  * Logistic Regression (`C`) – Regularization strength
  * Naive Bayes (`alpha`) – Laplace smoothing
  * Linear SVM (`C`) – Margin vs. classification error trade-off
* Evaluates models using validation F1-score
* Treats best-tuned versions as benchmark models
* Evaluates all benchmark models on the test set
* Selects the final best model based on **test F1-score**

### Outcome

The project compares multiple classical ML approaches for SMS spam detection and identifies the best-performing model based on validation and test performance.
