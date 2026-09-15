FROM python:3.12-slim

WORKDIR /app

COPY deployment/api/requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY deployment/api/main.py ./deployment/api/main.py
COPY models/adult_income_pipeline.joblib ./models/adult_income_pipeline.joblib

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "deployment.api.main:app", "--host", "0.0.0.0", "--port", "8000"]