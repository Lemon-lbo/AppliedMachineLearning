# AML Assignment 2 - SMS Spam Classification with Version Tracking

This repository extends SMS spam classification with **data versioning, experiment tracking, and model versioning**.

## Structure

* `prepare.ipynb` – Load, preprocess, split data, and **track dataset versions with DVC**
* `train.ipynb` – Train, tune, and **track models with MLflow**
* `raw_data.csv`, `train.csv`, `validation.csv`, `test.csv` – Raw and prepared datasets

### `prepare.ipynb`

* Loads `raw_data.csv`, cleans text, and splits into train/validation/test sets
* Tracks **data versions** with DVC (initial split + updated split with new random seed)
* Prints **target distribution** in each split for both versions
* **Optional**: Use Google Drive as remote storage

### `train.ipynb`

* Trains three models: Logistic Regression, Naive Bayes, Linear SVM
* Fine-tunes hyperparameters and evaluates with AUCPR
* Tracks experiments and model versions using MLflow
* Prints AUCPR for each benchmark model and selects the best

## Outcome

Demonstrates **data versioning**, **experiment tracking**, and **model versioning** while identifying the best classical ML model for SMS spam detection.

