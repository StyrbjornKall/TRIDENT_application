FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
COPY packages.txt .
RUN apt update && apt install -y libsm6 libxext6
RUN apt-get update && apt-get install -y libxrender-dev
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
ENTRYPOINT ["streamlit", "run", "app.py", "--server.fileWatcherType=none"]