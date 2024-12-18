#!/bin/bash

source /environment.sh

echo "Python executable: $(which python3)"
echo "Python version: $(python3 --version)"

pip3 freeze

# initialize launch file
dt-launchfile-init

## Set up CUDA environment
#export CUDA_HOME=/usr/local/cuda-10.2
#export PATH=/usr/local/cuda-10.2/bin:${PATH}
#export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64:/usr/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH}
#export CPATH=/usr/local/cuda-10.2/targets/aarch64-linux/include:/usr/local/cuda-10.2/include:${CPATH}

# launch subscriber
#dt-exec "${DT_REPO_PATH}"/packages/pycuda/build.sh
rosrun object_detection object_detection_node.py

# wait for app to end
dt-launchfile-join