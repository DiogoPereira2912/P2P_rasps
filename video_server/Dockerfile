FROM p2p-rpi

WORKDIR /app

ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app/video_server

COPY . .

CMD ["python3", "video-stream-server.py"]