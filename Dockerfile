FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

RUN pip install git+https://github.com/rmalpica/PyCSP.git

COPY pyproject.toml ./
RUN pip install .

COPY . .