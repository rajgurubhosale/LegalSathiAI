FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /bin/

WORKDIR /app
ENV PYTHONUNBUFFERED=1

COPY frontend/requirements.txt ./requirements.txt
RUN uv pip install --system --no-cache -r requirements.txt

COPY frontend ./frontend

EXPOSE 8501
CMD ["python", "-m", "streamlit", "run", "frontend/app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
