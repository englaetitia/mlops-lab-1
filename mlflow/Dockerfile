FROM python:3.11-slim

# Pinned to the client version used by the inference service and by training.
RUN pip install --no-cache-dir "mlflow==3.16.0"

RUN mkdir -p /mlflow-data

EXPOSE 5000

CMD ["mlflow", "server", \
    "--host", "0.0.0.0", \
    "--port", "5000", \
    "--backend-store-uri", "sqlite:////mlflow-data/mlflow.db", \
    "--artifacts-destination", "/mlflow-data/mlartifacts", \
    "--serve-artifacts", \
    "--allowed-hosts", "localhost,localhost:*,127.0.0.1,127.0.0.1:*,mlflow,mlflow:*"]