# see backend/Dockerfile for detail on some of the below commands
FROM python:3.12-slim AS base
COPY --from=ghcr.io/astral-sh/uv:0.7.9 /uv /uvx /bin/

ARG USER=frontend
RUN useradd -ms /bin/bash $USER
WORKDIR /app

COPY pyproject.toml uv.lock /app/
RUN uv sync

COPY app.py /app/
RUN chown -R $USER:$USER /app
USER $USER

EXPOSE 8501

# hide telemetry prompts for non-interactive envs
ENV STREAMLIT_TELEMETRY="0"

CMD [".venv/bin/streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
