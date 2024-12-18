#!/bin/bash

set -e

apt-get update
apt-get install -y --no-install-recommends \
    libopenblas-base \
    libopenmpi-dev \
    g++-8 \
    unzip
rm -rf /var/lib/apt/lists/*

# download TensorRT
echo "Downloading TensorRT v${TENSORRT_VERSION}..."
TENSORRT_WHEEL_NAME=tensorrt-${TENSORRT_VERSION}-cp38-cp38-linux_aarch64.whl
echo "Tensorrt wheel name: ${TENSORRT_WHEEL_NAME}"
TENSORRT_WHEEL_URL="https://duckietown-public-storage.s3.amazonaws.com/assets/python/wheels/${TENSORRT_WHEEL_NAME}"
wget -q "${TENSORRT_WHEEL_URL}" -O "/tmp/${TENSORRT_WHEEL_NAME}"
# install TensorRT
echo "Installing TensorRT v${TENSORRT_VERSION}..."
pip3 install "/tmp/${TENSORRT_WHEEL_NAME}"
rm "/tmp/${TENSORRT_WHEEL_NAME}"

# download PyCUDA
echo "Downloading PyCUDA v${PYCUDA_VERSION}..."
PYCUDA_WHEEL_NAME=pycuda-${PYCUDA_VERSION}-cp38-cp38-linux_aarch64.whl
echo "Pycuda wheel name: ${PYCUDA_WHEEL_NAME}"
PYCUDA_WHEEL_URL="https://duckietown-public-storage.s3.amazonaws.com/assets/python/wheels/${PYCUDA_WHEEL_NAME}"
wget -q "${PYCUDA_WHEEL_URL}" -O "/tmp/${PYCUDA_WHEEL_NAME}"
# install PyCUDA
echo "Installing PyCUDA v${PYCUDA_VERSION}..."
pip3 install "/tmp/${PYCUDA_WHEEL_NAME}"
rm "/tmp/${PYCUDA_WHEEL_NAME}"

# clean
pip3 uninstall -y dataclasses

python3 -c "import sys; print('Python paths after install:', sys.path)" && \
ls -la /usr/local/lib/python3.8/dist-packages/ && \
python3 -c "import pkg_resources; print([p for p in pkg_resources.working_set if 'pycuda' in p.key])"