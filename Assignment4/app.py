import joblib
from flask import Flask, request, jsonify
from score import score

import os

app = Flask(__name__)

MODEL_PATH = "best_model.joblib"
DEFAULT_THRESHOLD = 0.5

model = joblib.load(MODEL_PATH)


@app.route("/score", methods=["POST"])
def score_endpoint():
    """
    POST /score
    JSON body  : {"text": "<sms message>", "threshold": 0.5}
    JSON response: {"prediction": true/false, "propensity": 0.97}
    """
    data = request.get_json(force=True)

    if "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400

    text = data["text"]
    threshold = float(data.get("threshold", DEFAULT_THRESHOLD))

    prediction, propensity = score(text, model, threshold)

    return jsonify({
        "prediction": prediction,
        "propensity": propensity
    })


if __name__ == "__main__":
    port = int(os.environ.get("FLASK_PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
