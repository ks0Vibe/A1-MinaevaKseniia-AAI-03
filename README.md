# Adult Income ML Pipeline

The project preprocesses the UCI Adult dataset, trains a logistic regression model, and deploys a FastAPI API and a Streamlit app in separate Docker containers. Airflow runs the full pipeline every five minutes.

## Requirements

- Python 3.11+
- Docker with Docker Compose
- UCI Adult files `adult.data` and `adult.test`

## Setup

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdir -p data/raw/adult
```

Place the dataset files here:

```text
data/raw/adult/adult.data
data/raw/adult/adult.test
```

## Manual run

```bash
python src/preprocess.py
python src/train.py
docker compose -f deployment/docker-compose.yml up --build -d
```

Open:

- Streamlit app: http://localhost:8501
- FastAPI docs: http://localhost:8000/docs

Stop the containers:

```bash
docker compose -f deployment/docker-compose.yml down
```

## Airflow pipeline

Keep the virtual environment active and Docker running, then execute:

```bash
export AIRFLOW_HOME="$PWD/services/airflow"
airflow standalone
```

Open http://localhost:8080 and enable the `adult_income_ml_pipeline` DAG. It preprocesses the data, trains the model, and rebuilds the deployment every five minutes.
