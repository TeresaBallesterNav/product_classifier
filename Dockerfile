# Base Python image
FROM python:3.12-slim

# Setting working directory
WORKDIR /app

# Requirements and installation
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App
COPY main.py .
COPY src/ ./src
COPY models/ ./models

# Run FastAPI with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]