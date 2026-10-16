

from datetime import datetime, timedelta

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG

PROJECT_DIR = "/opt/project"


def run_module(module: str) -> str:

    return f"cd {PROJECT_DIR} && uv run --frozen --no-dev python -m projet_home.{module}"


with DAG(
    dag_id="rossmann_pipeline",
    description="train, predict, drift",
    start_date=datetime(2026, 1, 1),
    schedule="@weekly",        
    catchup=False,              
    default_args={"retries": 1, "retry_delay": timedelta(minutes=5)},
    tags=["rossmann"],
) as dag:

    train = BashOperator(
        task_id="train",
        bash_command=run_module("train"),
        execution_timeout=timedelta(hours=1),
    )

    predict = BashOperator(
        task_id="predict",
        bash_command=run_module("predict"),
        execution_timeout=timedelta(minutes=30),
    )

    drift = BashOperator(
        task_id="drift",
        bash_command=run_module("drift"),
        execution_timeout=timedelta(minutes=30),
    )

    train >> predict >> drift