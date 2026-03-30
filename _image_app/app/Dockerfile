# Production Dockerfile for the Dash app
# Builds a minimal image and runs via Gunicorn (WSGI) for production

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install system deps required by some packages (keep minimal)
RUN apt-get update \
    # && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && pip check || true

# Create non-root user early so we can COPY files with correct owner
RUN useradd -m appuser

# Copy application code and set ownership during copy to avoid slow chown -R
COPY --chown=appuser:appuser . ./

# Switch to non-root user
USER appuser

EXPOSE 8050

# Lightweight healthcheck using builtin urllib; no extra packages
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
  CMD ["python","-c","import urllib.request,sys; r=urllib.request.urlopen('http://127.0.0.1:8050'); sys.exit(0 if r.getcode()==200 else 1)"]

# Run the app via Gunicorn WSGI server (launch_dashboard exposes 'server')
CMD ["python", "launch_dashboard.py"]
