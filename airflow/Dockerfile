FROM apache/airflow:3.3.2

# LightGBM a besoin de cette bibliothèque système
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
USER airflow

# uv : installe l'environnement du projet, séparé de celui d'Airflow
RUN pip install --no-cache-dir apache-airflow==${AIRFLOW_VERSION} uv

# L'environnement du projet vit dans le conteneur, pas dans le dossier partagé
# (le .venv de Windows n'est pas utilisable sous Linux)
ENV UV_PROJECT_ENVIRONMENT=/opt/airflow/project-venv
ENV UV_LINK_MODE=copy
RUN mkdir -p /opt/airflow/project-venv