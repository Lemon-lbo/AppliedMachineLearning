"""
Unit tests for score() and integration test for Flask /score endpoint.
Run with:
    pytest test.py -v --cov=score --cov=app --cov-report=term-missing | tee coverage.txt
"""
import os
import time
import signal
import subprocess

import joblib
import requests
import pytest

from score import score

MODEL_PATH = "best_model.joblib"
FLASK_URL = "http://127.0.0.1:5002/score"
IMAGE_NAME = "spam-flask-app"
HOST_PORT = "5001"  # host port we used with docker run -p 5001:5000
DOCKER_FLASK_URL = f"http://127.0.0.1:{HOST_PORT}/score"

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def model():
    return joblib.load(MODEL_PATH)


# ---------------------------------------------------------------------------
# Unit tests – score()
# ---------------------------------------------------------------------------

class TestScore:

    def test_smoke(self, model):
        """Smoke test: function runs without crashing."""
        result = score("Congratulations! You won a free prize!", model, 0.5)
        assert result is not None

    def test_output_format(self, model):
        """Format test: returns a tuple of (bool, float)."""
        prediction, propensity = score("Hello, how are you?", model, 0.5)
        assert isinstance(prediction, bool)
        assert isinstance(propensity, float)

    def test_prediction_is_binary(self, model):
        """Sanity check: prediction is True or False."""
        prediction, _ = score("Win cash now!", model, 0.5)
        assert prediction in (True, False)

    def test_propensity_range(self, model):
        """Sanity check: propensity is between 0 and 1."""
        _, propensity = score("Call me when free", model, 0.5)
        assert 0.0 <= propensity <= 1.0

    def test_threshold_zero_always_spam(self, model):
        """Edge case: threshold=0 → prediction always True (spam)."""
        prediction, _ = score("See you tomorrow", model, 0.0)
        assert prediction is True

    def test_threshold_one_always_ham(self, model):
        """Edge case: threshold=1 → prediction always False (ham)."""
        prediction, _ = score("FREE CASH WIN NOW CALL 0800", model, 1.0)
        assert prediction is False

    def test_obvious_spam(self, model):
        """Typical input: obvious spam text → prediction True."""
        spam_text = (
            "URGENT! You have won a 1 week FREE membership in our "
            "prize reward scheme! To claim call 09061743811"
        )
        prediction, _ = score(spam_text, model, 0.5)
        assert prediction is True

    def test_obvious_ham(self, model):
        """Typical input: obvious ham text → prediction False."""
        ham_text = "Hey, are you coming to dinner tonight?"
        prediction, _ = score(ham_text, model, 0.5)
        assert prediction is False


# ---------------------------------------------------------------------------
# Integration test – Flask /score endpoint
# ---------------------------------------------------------------------------

class TestFlask:

    @pytest.fixture(autouse=True, scope="class")
    def flask_app(self):
        """Launch and tear down the Flask app around the test class."""
        env = os.environ.copy()
        env["FLASK_PORT"] = "5002"

        proc = subprocess.Popen(
            ["python", "app.py"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
        )
        time.sleep(2)          # wait for server to start
        yield proc
        proc.send_signal(signal.SIGTERM)
        proc.wait()

    def test_flask_smoke(self):
        """Flask smoke test: endpoint responds to a POST request."""
        resp = requests.post(FLASK_URL, json={"text": "Hello there"})
        assert resp.status_code == 200

    def test_flask_response_format(self):
        """Flask format test: response contains prediction and propensity keys."""
        resp = requests.post(FLASK_URL, json={"text": "Win a free iPhone now!"})
        data = resp.json()
        assert "prediction" in data
        assert "propensity" in data

    def test_flask_prediction_type(self):
        """Flask type test: prediction is bool, propensity is float."""
        resp = requests.post(FLASK_URL, json={"text": "Are you free tonight?"})
        data = resp.json()
        assert isinstance(data["prediction"], bool)
        assert isinstance(data["propensity"], float)

    def test_flask_spam_detection(self):
        """Flask typical input: known spam text → prediction True."""
        spam_text = "Congratulations! You have been selected for a cash prize. Call now!"
        resp = requests.post(FLASK_URL, json={"text": spam_text})
        assert resp.json()["prediction"] is True

    def test_flask_ham_detection(self):
        """Flask typical input: known ham text → prediction False."""
        resp = requests.post(FLASK_URL, json={"text": "See you at the library"})
        assert resp.json()["prediction"] is False

    def test_flask_custom_threshold(self):
        """Flask threshold test: threshold=1 forces prediction False."""
        resp = requests.post(
            FLASK_URL,
            json={"text": "FREE prize winner CALL NOW", "threshold": 1.0}
        )
        assert resp.json()["prediction"] is False

    def test_flask_missing_text_returns_400(self):
        """Flask error handling: missing text field → 400 status."""
        resp = requests.post(FLASK_URL, json={"threshold": 0.5})
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Docker container test – Flask /score endpoint inside Docker
# ---------------------------------------------------------------------------

class TestDocker:
    @pytest.fixture(scope="class", autouse=True)
    def docker_container(self, request):
        """
        Build the Docker image and run the container for this test class.
        """
        # Build image: docker build -t spam-flask-app .
        build = subprocess.run(
            ["docker", "build", "-t", IMAGE_NAME, "."],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if build.returncode != 0:
            pytest.skip(f"Docker build failed or Docker not available:\n{build.stderr}")

        # Run container: docker run --rm -p 5001:5000 spam-flask-app
        run_proc = subprocess.Popen(
            [
                "docker",
                "run",
                "--rm",
                "-p",
                f"{HOST_PORT}:5000",
                IMAGE_NAME,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Wait a bit for the server to start
        time.sleep(5)

        def fin():
            # Stop container process when tests are done
            run_proc.terminate()
            try:
                run_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                run_proc.kill()

        request.addfinalizer(fin)
        return run_proc

    def test_docker_score_endpoint(self, docker_container):
        """
        Send a request to the Dockerized Flask /score endpoint and check response.
        """
        resp = requests.post(
            DOCKER_FLASK_URL,
            json={"text": "Congratulations! You won a cash prize!", "threshold": 0.5},
            timeout=5,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "prediction" in data
        assert "propensity" in data
        assert isinstance(data["prediction"], bool)
        assert isinstance(data["propensity"], float)
