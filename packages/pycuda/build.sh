#!/usr/bin/env bash

#
#   NOTE: This script is based on the instructions from the official website:
#       https://docs.cupy.dev/en/stable/install.html#installing-cupy-from-source
#


echo ${CATKIN_WS}


OUTPUT_DIR=/out
SCRIPTPATH="$(
    cd "$(dirname "$0")" >/dev/null 2>&1
    pwd -P
)"

# check volume
mountpoint -q "${OUTPUT_DIR}"
if [ $? -ne 0 ]; then
  echo "ERROR: The path '${OUTPUT_DIR}' is not a VOLUME. The resulting artefacts would be lost.
  Mount an external directory to it and retry."
  exit 1
fi

set -ex

cd "${SCRIPTPATH}"

ls /usr/local

pip3 install numpy
python3 configure.py --cuda-root=/usr/local/cuda-10.2

echo $PATH
echo $LD_LIBRARY_PATH

mkdir dist/
python3 setup.py bdist_wheel
cp -R dist/* /out/