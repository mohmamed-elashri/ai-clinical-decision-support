FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.lock ./
RUN pip install --no-cache-dir -r requirements.lock

COPY . .
RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app
USER appuser

ENV APP_ENV=production \
    APP_DEBUG=0 \
    PORT=7860 \
    FLASK_PORT=5000 \
    GRADIO_PORT=7860 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    FLASK_API_URL=http://127.0.0.1:5000 \
    ALLOW_EXTRACTIVE_FALLBACK=0

EXPOSE 7860
CMD ["python", "serve.py"]
