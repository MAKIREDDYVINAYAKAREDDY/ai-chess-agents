FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        stockfish \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application
COPY app.py .
COPY chess_ai.py .
COPY chess_analysis.py .
COPY chess_env_v4.py .
COPY chess_env_masked.py .

COPY templates ./templates
COPY static ./static

# PPO model
COPY chess_ai_agent_v2_masked.zip .

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "120", "app:app"]
