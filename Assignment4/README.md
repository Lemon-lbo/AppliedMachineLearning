# Assignment 4: Containerization and Continuous Integration

This assignment adds Docker containerization and a simple continuous integration step on top of the spam classifier Flask app from Assignment 3.

## Repository Structure

```text
Assignment 4/
├── Dockerfile
├── app.py
├── score.py
├── test.py
├── test_colab.py
├── best_model.joblib
├── coverage.txt
├── README.md
├── hooks/
│   └── pre-commit.sample
├── Model Training/
│   ├── train.csv
│   ├── validation.csv
│   ├── test.csv
│   ├── updated_train.ipynb
│   └── mlflow.db
└── main.ipynb
```

## Files

- `Dockerfile` – builds a container for the Flask app using `app.py`, `score.py`, and `best_model.joblib`.
- `app.py` – Flask API exposing `POST /score` for spam prediction.
- `score.py` – scoring function that returns a boolean prediction and a propensity score.
- `test.py` – unit tests for `score()`, integration tests for `/score`, and a Docker container test.
- `test_colab.py` – alternative test runner used in the Colab environment.
- `coverage.txt` – pytest coverage report for the tests in `test.py`.
- `hooks/pre-commit.sample` – pre-commit hook script that runs `pytest test.py` before each commit.
- `Model Training/` – training notebook, MLflow artifacts, and train/validation/test splits from earlier assignments.

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
pip install pytest pytest-cov flask scikit-learn joblib requests
```

## Docker

Build the Docker image:

```bash
docker build -t spam-flask-app .
```

Run the container (host port 5001 → container port 5000):

```bash
docker run --rm -p 5001:5000 spam-flask-app
```

Test the running container:

```bash
curl -X POST http://127.0.0.1:5001/score \
     -H "Content-Type: application/json" \
     -d '{"text": "Win a free iPhone!", "threshold": 0.5}'
```

## Tests and Coverage

Run all tests with coverage and write the report to `coverage.txt`:

```bash
pytest test.py -v --cov=score --cov=app --cov-report=term-missing | tee coverage.txt
```

## Pre-commit Hook

To enable the pre-commit hook locally:

```bash
cp hooks/pre-commit.sample .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

This will run `pytest test.py` automatically on each commit to the local repository.
