# TODO: could alpine suffice here? (and in frontend)
FROM python:3.12-slim AS base
# get uv binary direct - see https://docs.astral.sh/uv/guides/integration/docker/#installing-uv
COPY --from=ghcr.io/astral-sh/uv:0.7.9 /uv /uvx /bin/

# install system libs that lightweight wheels sometimes lack
RUN apt-get update && \
  apt-get install -y --no-install-recommends \
  build-essential && \
  rm -rf /var/lib/apt/lists/*

# create an unprivileged user (basic container hygiene) and set working directory
ARG USER=api
RUN useradd -ms /bin/bash $USER
WORKDIR /app

# copy what we need to install Python deps first (to make best use of Docker caching), and sync
COPY pyproject.toml uv.lock /app/
# TODO: define dev dependencies to reduce deps installed in container
RUN uv sync

# copy relevant code and create models directory
COPY api.py data.py model.py search.py /app/
RUN mkdir -p /app/models

# give new user ownership of the workdir, and switch
RUN chown -R $USER:$USER /app
USER $USER

# expose the port FastAPI/gunicorn will listen on (this line is just documentation, it does nothing)
EXPOSE 8000

# start the service (call gunicorn server binary directly with Uvicorn worker on port 8000)
CMD [".venv/bin/gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "1", "--bind", "0.0.0.0:8000", "api:api"]
