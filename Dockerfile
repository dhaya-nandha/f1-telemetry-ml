# ─── STAGE 1: BUILD ENVIRONMENT ───────────────────────────────────────
FROM python:3.10-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install dependencies into a localized wheels/virtual layer
RUN pip install --no-cache-dir --user -r requirements.txt

# ─── STAGE 2: FINAL RUNTIME ───────────────────────────────────────────
FROM python:3.10-slim AS runner

WORKDIR /app

# Install curl strictly for the HEALTHCHECK
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed dependencies from builder stage
COPY --from=builder /root/.local /root/.local
COPY --from=builder /app/requirements.txt .

# Copy project directories
COPY src/ ./src/
COPY models/ ./models/

# Append the local user bin to path so streamlit is executable
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

EXPOSE 8501

# Validated healthcheck using curl
HEALTHCHECK --interval=30s --timeout=3s \
  CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]