FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1

# Install system dependencies if needed (kept minimal)
COPY pyproject.toml requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source and install package (so CLI/entrypoints work)
COPY . .
RUN pip install --no-cache-dir .

EXPOSE 8000
CMD ["uvicorn", "triagerl.api.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
