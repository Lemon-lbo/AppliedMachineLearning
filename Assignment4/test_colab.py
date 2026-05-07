import os, time, threading, joblib, requests, pytest
from score import score
from app import app as flask_app

MODEL_PATH = 'best_model.joblib'

@pytest.fixture(scope='module')
def model():
    return joblib.load(MODEL_PATH)

@pytest.fixture(scope='module')
def flask_url():
    t = threading.Thread(
        target=lambda: flask_app.run(host='0.0.0.0', port=5000, use_reloader=False),
        daemon=True)
    t.start()
    time.sleep(2)
    return 'http://127.0.0.1:5000'

class TestScore:

    def test_smoke(self, model):
        assert score('Free prize!', model, 0.5) is not None

    def test_output_format(self, model):
        p, s = score('Hello there', model, 0.5)
        assert isinstance(p, bool) and isinstance(s, float)

    def test_prediction_is_binary(self, model):
        p, _ = score('Win cash now!', model, 0.5)
        assert p in (True, False)

    def test_propensity_range(self, model):
        _, s = score('Call me when free', model, 0.5)
        assert 0.0 <= s <= 1.0

    def test_threshold_zero_always_spam(self, model):
        p, _ = score('See you tomorrow', model, 0.0)
        assert p is True

    def test_threshold_one_always_ham(self, model):
        p, _ = score('FREE CASH WIN NOW CALL 0800', model, 1.0)
        assert p is False

    def test_obvious_spam(self, model):
        p, _ = score('URGENT you won FREE membership call 09061743811', model, 0.5)
        assert p is True

    def test_obvious_ham(self, model):
        p, _ = score('Hey are you coming to dinner tonight?', model, 0.5)
        assert p is False

class TestFlask:

    def test_flask_smoke(self, flask_url):
        r = requests.post(flask_url + '/score', json={'text': 'Hello'})
        assert r.status_code == 200

    def test_flask_response_format(self, flask_url):
        d = requests.post(flask_url + '/score', json={'text': 'Win free iPhone!'}).json()
        assert 'prediction' in d and 'propensity' in d

    def test_flask_prediction_type(self, flask_url):
        d = requests.post(flask_url + '/score', json={'text': 'Are you free tonight?'}).json()
        assert isinstance(d['prediction'], bool) and isinstance(d['propensity'], float)

    def test_flask_spam_detection(self, flask_url):
        r = requests.post(flask_url + '/score', json={'text': 'Congratulations cash prize call now!'})
        assert r.json()['prediction'] is True

    def test_flask_ham_detection(self, flask_url):
        r = requests.post(flask_url + '/score', json={'text': 'See you at the library'})
        assert r.json()['prediction'] is False

    def test_flask_custom_threshold(self, flask_url):
        r = requests.post(flask_url + '/score',
                          json={'text': 'FREE prize CALL NOW', 'threshold': 1.0})
        assert r.json()['prediction'] is False

    def test_flask_missing_text_returns_400(self, flask_url):
        r = requests.post(flask_url + '/score', json={'threshold': 0.5})
        assert r.status_code == 400
