#!/bin/bash

# Set up TAPPAS workspace (adjust this if you're mounting another workspace)
export TAPPAS_WORKSPACE="/usr/lib/aarch64-linux-gnu/hailo/tappas"
echo "TAPPAS_WORKSPACE set to $TAPPAS_WORKSPACE"

# Set TAPPAS_POST_PROC_DIR based on workspace
export TAPPAS_POST_PROC_DIR="${TAPPAS_WORKSPACE}/post_processes/"
echo "TAPPAS_POST_PROC_DIR set to $TAPPAS_POST_PROC_DIR"

# Get and export device architecture
output=$(hailortcli fw-control identify | tr -d '\0')
device_arch=$(echo "$output" | grep "Device Architecture" | awk -F": " '{print $2}')

if [ -z "$device_arch" ]; then
    echo "Error: Device Architecture not found. Please check the connection to the device."
    return 1  # or exit 1 if running as a script
fi

export DEVICE_ARCHITECTURE="$device_arch"
echo "DEVICE_ARCHITECTURE is set to: $DEVICE_ARCHITECTURE"

export XDG_RUNTIME_DIR="/run/user/1000"
echo "XDG_RUNTIME_DIR is set to: $XDG_RUNTIME_DIR"