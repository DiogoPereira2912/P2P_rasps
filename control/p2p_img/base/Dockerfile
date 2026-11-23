# # # syntax=docker/dockerfile:1-labs

# # # Base image for using Hailo-8 on Raspberry Pi

# FROM debian:bookworm-slim

# RUN apt-get update && apt-get install -y software-properties-common wget gnupg

# RUN echo "# See sources.lists.d directory" > /etc/apt/sources.list && \
#     apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 82B129927FA3303E && \
#     apt-add-repository -y -S deb http://archive.raspberrypi.com/debian/ bookworm main

# RUN apt-get update && apt-get install -y \
#     python3 python3-pip python3-dev python3-setuptools \
#     gcc-12 g++-12 python3-venv \
#     ffmpeg x11-utils python3-opencv libopencv-dev \
#     libcairo2-dev libgirepository1.0-dev libgstreamer1.0-dev \
#     libgstreamer-plugins-base1.0-dev gstreamer1.0-plugins-base \
#     gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-libav \
#     gstreamer1.0-tools gstreamer1.0-x gstreamer1.0-libcamera \
#     rpicam-apps hailo-tappas-core hailo-all \
#     libgl1-mesa-glx libglib2.0-0 rsync git cmake meson \
#     rapidjson-dev pkg-config \
#     sqlite3

# COPY requirements.txt .

# RUN python3 -m pip install --no-cache-dir --break-system-packages --upgrade pip
# RUN python3 -m pip install --break-system-packages -r requirements.txt

# # WORKDIR /opt/install
# # COPY install.sh requirements.txt ./
# # RUN chmod +x install.sh && ./install.sh

# # SHELL ["/bin/bash", "-c"]
# # CMD ["bash"]

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y software-properties-common wget gnupg \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN echo "# See sources.lists.d directory" > /etc/apt/sources.list && \
    apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 82B129927FA3303E && \
    apt-add-repository -y -S deb http://archive.raspberrypi.com/debian/ bookworm main

# Instala pacotes e limpa a cache do APT na mesma camada
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev python3-setuptools \
    gcc-12 g++-12 python3-venv \
    ffmpeg x11-utils python3-opencv libopencv-dev \
    libcairo2-dev libgirepository1.0-dev libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-libav \
    gstreamer1.0-tools gstreamer1.0-x gstreamer1.0-libcamera \
    rpicam-apps hailo-tappas-core hailo-all \
    libgl1-mesa-glx libglib2.0-0 rsync git cmake meson \
    rapidjson-dev pkg-config \
    sqlite3 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python3 -m pip install --no-cache-dir --break-system-packages --upgrade pip

RUN python3 -m pip install --no-cache-dir --break-system-packages torch torchvision --index-url https://download.pytorch.org/whl/cpu

RUN python3 -m pip install --no-cache-dir --break-system-packages -r requirements.txt

RUN python3 -m pip install --no-cache-dir --break-system-packages --no-build-isolation "git+https://github.com/hailo-ai/hailo-apps-infra.git@25.3.1"