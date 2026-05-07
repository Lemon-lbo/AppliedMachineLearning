FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir flask scikit-learn joblib requests

COPY app.py score.py best_model.joblib ./

EXPOSE 5000

CMD ["python", "app.py"]