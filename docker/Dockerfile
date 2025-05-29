# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libffi-dev \
    libssl-dev \
    wget \
    curl \
    git \
    ffmpeg \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application
COPY . .

# Create necessary directories
RUN mkdir -p vector_store data logs /var/log/supervisor

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Create supervisor configuration
RUN echo '[supervisord]\n\
nodaemon=true\n\
logfile=/var/log/supervisor/supervisord.log\n\
pidfile=/var/run/supervisord.pid\n\
\n\
[program:fastapi]\n\
command=python -m uvicorn services.api_service:app --host 0.0.0.0 --port 8000 --workers 1\n\
directory=/app\n\
autostart=true\n\
autorestart=true\n\
stderr_logfile=/var/log/supervisor/fastapi_err.log\n\
stdout_logfile=/var/log/supervisor/fastapi_out.log\n\
environment=PYTHONPATH="/app"\n\
\n\
[program:streamlit]\n\
command=python -m streamlit run main.py --server.port 8501 --server.address 0.0.0.0 --server.headless true --server.runOnSave false\n\
directory=/app\n\
autostart=true\n\
autorestart=true\n\
stderr_logfile=/var/log/supervisor/streamlit_err.log\n\
stdout_logfile=/var/log/supervisor/streamlit_out.log\n\
environment=PYTHONPATH="/app"\n\
' > /etc/supervisor/conf.d/supervisord.conf

# Expose ports
EXPOSE 8000 8501

# Use supervisor to manage both services
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]