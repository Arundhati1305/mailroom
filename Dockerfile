FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
ENV PORT=8000
# one worker: in-process dedupe of concurrent model work relies on it
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT} --workers 1 --timeout-keep-alive 75
