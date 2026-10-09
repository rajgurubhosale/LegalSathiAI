FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /bin/

WORKDIR /app
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN uv pip install --system --no-cache torch --index-url https://download.pytorch.org/whl/cpu
COPY src/backend/requirements.txt ./requirements.txt
RUN uv pip install --system --no-cache -r requirements.txt

COPY src ./src
RUN touch .project-root

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "src.backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
