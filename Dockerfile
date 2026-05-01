FROM python:3.10

WORKDIR /app

COPY . /app

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Expose port for HF
EXPOSE 7860

# Run app
CMD ["python", "app.py"]
