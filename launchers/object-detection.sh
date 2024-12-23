#!/bin/bash

source /environment.sh

echo "Python executable: $(which python3)"
echo "Python version: $(python3 --version)"

pip3 freeze

# initialize launch file
dt-launchfile-init

## Set up CUDA environment

# launch subscriber
rosrun object_detection object_detection_node.py

# wait for app to end
dt-launchfile-join