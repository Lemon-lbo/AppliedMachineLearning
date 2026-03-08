# Assignment 3: Testing and Model Serving

## Overview

This assignment implements testing and serving for a spam classification model. A scoring function is created to generate predictions from text input, and the functionality is verified using unit tests and integration tests. The model is served through a Flask API, and test coverage is evaluated using `pytest` and `pytest-cov`.

## Repository Structure

```
assignment3/
│
├── app.py
├── score.py
├── test.py
├── test_colab.py
│
├── updated_train.ipynb
├── main.ipynb
│
├── best_model.joblib
├── coverage.txt
│
└── data/
    ├── train.csv
    ├── test.csv
    └── validation.csv
```

The `data/` folder contains the datasets used for training and evaluating the model.

## File Descriptions

- **score.py**
Implements the `score()` function that takes a text input, a trained sklearn model, and a threshold, and returns a binary prediction (0 or 1) and a probability score (propensity).

- **app.py**
Creates a Flask API with a `/score` POST endpoint that receives text input and returns prediction and propensity in JSON format.

- **test.py**
Contains pytest unit tests for the scoring function and integration tests that start the Flask app, send requests to the `/score` endpoint, and validate the responses.

- **test_colab.py**
A modified version of the tests adapted to run correctly in the Google Colab environment.

- **updated_train.ipynb**
An updated version of the `train.ipynb` notebook from Assignment 2 that trains models and saves the best model in `joblib` format.

- **main.ipynb**
Runs the full workflow in Colab: uploads files, runs the test suite, and generates the coverage report.

- **coverage.txt**
Contains the coverage report produced by `pytest-cov`.

## Running Tests

Install dependencies:

```
pip install pytest pytest-cov flask scikit-learn joblib requests
```

Run tests and generate coverage:

```
pytest --cov=. > coverage.txt
```



