# syntax=docker/dockerfile:1-labs
 
FROM python:3.11-slim

WORKDIR /app 

COPY requirements.txt .

RUN apt-get update && apt-get install -y rsync sshpass openssh-client && rm -rf /var/lib/apt/lists/*
RUN apt-get update && pip3 install --no-cache-dir -r requirements.txt

COPY . .

CMD ["streamlit", "run", "gossip_ui.py", "--server.address=0.0.0.0"]