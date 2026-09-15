from datetime import datetime
from pathlib import Path
import sys

from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT_ROOT = Path(__file__).resolve().parents[3]

PYTHON_EXECUTABLE = sys.executable

PREPROCESS_SCRIPT = PROJECT_ROOT / "src" / "preprocess.py"
TRAIN_SCRIPT = PROJECT_ROOT / "src" / "train.py"

DOCKER_COMPOSE_FILE = (
    PROJECT_ROOT
    / "deployment"
    / "docker-compose.yml"
)



with DAG(
    dag_id="adult_income_ml_pipeline",

    description=(
        "Automated Adult Income ML pipeline: "
        "data preprocessing, model training and deployment."
    ),

    start_date=datetime(2026, 9, 15),

    schedule="*/5 * * * *",

    catchup=False,

    max_active_runs=1,

    tags=[
        "pmldl",
        "mlops",
        "adult-income",
    ],
) as dag:


    preprocess_data = BashOperator(
        task_id="preprocess_data",

        bash_command=(
            f'cd "{PROJECT_ROOT}" && '
            f'"{PYTHON_EXECUTABLE}" '
            f'"{PREPROCESS_SCRIPT}"'
        ),
    )


    train_model = BashOperator(
        task_id="train_model",

        bash_command=(
            f'cd "{PROJECT_ROOT}" && '
            f'"{PYTHON_EXECUTABLE}" '
            f'"{TRAIN_SCRIPT}"'
        ),
    )


    deploy_services = BashOperator(
        task_id="deploy_services",

        bash_command=(
            f'cd "{PROJECT_ROOT}" && '
            f'docker compose '
            f'-f "{DOCKER_COMPOSE_FILE}" '
            f'up --build -d'
        ),
    )


    preprocess_data >> train_model >> deploy_services