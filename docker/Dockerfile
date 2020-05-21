FROM pytorch/pytorch:latest

RUN apt-get update && apt-get install -y git \
    && pip install --upgrade pip \
        pylint \
        autopep8 \
        matplotlib

WORKDIR /work