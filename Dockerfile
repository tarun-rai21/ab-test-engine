# Two images from one Dockerfile, each with only what it needs:
#   --target api  FastAPI service (requirements-api.txt: no Streamlit/matplotlib)
#   --target app  Streamlit report app (requirements.txt: API deps + UI)
# Neither installs development tools (requirements-dev.txt).
FROM python:3.13-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg \
    DATABASE_URL=sqlite:////data/ab_test_engine.db

WORKDIR /app

# Run as an unprivileged user; /data holds the SQLite database (a volume).
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /data \
    && chown appuser /data


FROM base AS api

COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

COPY . .
USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]


FROM base AS app

COPY requirements-api.txt requirements.txt ./
RUN pip install -r requirements.txt

COPY . .
USER appuser
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4)"

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
