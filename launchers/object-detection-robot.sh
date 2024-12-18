#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# launch subscriber
#rosrun object_detection object_detection_node.py

dt-exec "${DT_REPO_PATH}"/packages/pycuda/build.sh

# wait for app to end
dt-launchfile-join