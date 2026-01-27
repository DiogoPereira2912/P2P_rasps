# syntax=docker/dockerfile:1-labs
 
FROM hailo-rpi

WORKDIR /app 

RUN apt-get update 

COPY . .

COPY setup_vars.sh /opt/setup_vars.sh
RUN chmod +x /opt/setup_vars.sh

EXPOSE 80

CMD ["bash", "-c", "source /opt/setup_vars.sh && python3 -u yolo_pose.py"]