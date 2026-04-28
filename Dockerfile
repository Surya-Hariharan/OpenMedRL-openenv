FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "medical_triage_env.env:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
