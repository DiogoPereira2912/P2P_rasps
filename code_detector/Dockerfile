# syntax=docker/dockerfile:1-labs
 
FROM p2p-rpi

WORKDIR /app

ENV PYTHONUNBUFFERED=1

RUN python3 -m pip install --break-system-packages mmh3 pymmh3

COPY . .

CMD ["bash", "-c", "chmod +x setup_vars.sh && source ./setup_vars.sh && python3 -u yolo_pose.py"]