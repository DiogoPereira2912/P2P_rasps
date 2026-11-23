#!/bin/bash
set -e

# Run this to generate a base image using DeGirum Models
echo "[Base] Installing Python dependencies..."
python3 -m pip install --break-system-packages -r requirements.txt

TAG="25.3.1"
echo "[Base] Installing hailo-apps-infra from tag ${TAG}..."
python3 -m pip install --no-cache-dir --break-system-packages --no-build-isolation "git+https://github.com/hailo-ai/hailo-apps-infra.git@$TAG"