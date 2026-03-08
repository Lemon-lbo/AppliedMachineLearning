import joblib
import numpy as np
import sklearn


def score(text: str, model: sklearn.base.BaseEstimator, threshold: float):
    """
    Score a single text string using a trained sklearn pipeline.

    Parameters
    ----------
    text      : raw SMS string
    model     : fitted sklearn Pipeline (TfidfVectorizer + classifier)
    threshold : decision threshold in [0, 1]

    Returns
    -------
    prediction  : bool  – True if spam (1), False if ham (0)
    propensity  : float – probability / confidence score in [0, 1]
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not (0.0 <= threshold <= 1.0):
        raise ValueError("threshold must be between 0 and 1")

    # Get propensity score
    if hasattr(model, "predict_proba"):
        propensity = float(model.predict_proba([text])[0][1])
    else:
        # LinearSVC uses decision_function; map to [0,1] via sigmoid
        raw = model.decision_function([text])[0]
        propensity = float(1 / (1 + np.exp(-raw)))

    prediction = bool(propensity >= threshold)
    return prediction, propensity
