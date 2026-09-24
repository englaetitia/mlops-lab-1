FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir --retries 10 --timeout 120 streamlit requests

COPY app.py .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", \
    "--server.address=0.0.0.0", \
    "--server.port=8501", \
    "--server.headless=true", \
    "--browser.gatherUsageStats=false"]