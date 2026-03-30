# Production Dockerfile for the Dash app
# Images are baked in via a compressed archive for a single portable image.
#
# PREP (once, or whenever images change):
#   tar czf images_cache.tar.gz -C images_cache .
#
# BUILD:
#   docker build -t sueltas-app .
#
# RUN:
#   docker run -d -p 8050:8050 sueltas-app

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install system deps required by some packages (keep minimal)
RUN apt-get update \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (layer caching — deps change less than code)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && pip check || true

# Create non-root user
RUN useradd -m appuser

# Copy application code (COPY --chown avoids slow recursive chown)
COPY --chown=appuser:appuser . ./

# Extract images from compressed archive (ADD auto-extracts .tar.gz)
ADD --chown=appuser:appuser images_cache.tar.gz ./images_cache/

USER appuser

EXPOSE 8050

# Lightweight healthcheck using builtin urllib; no extra packages
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
  CMD ["python","-c","import urllib.request,sys; r=urllib.request.urlopen('http://127.0.0.1:8050'); sys.exit(0 if r.getcode()==200 else 1)"]

CMD ["python", "launch_dashboard.py"]
