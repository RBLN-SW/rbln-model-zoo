#!/bin/bash

OUTPUT_DIR="output"

if [ ! -d ${OUTPUT_DIR} ]; then
  mkdir ${OUTPUT_DIR}
fi

pushd ${OUTPUT_DIR}

echo "=================================================================="
echo "= Step 1. YOLOv8 model compile                                 ="
echo "=================================================================="

DIR="../material/yolov8"
MODEL="yolov8l"
MODEL_PATH="${DIR}/${MODEL}.rbln"
LABEL_DST="../material"
LABEL_SRC="ultralytics/ultralytics/cfg/datasets"
LABEL_FILE="coco128.yaml"
if [ ! -f ${DIR}/${MODEL}.rbln ] || [ ! -f ${LABEL_DST}/${LABEL_FILE} ]; then
  pip3 install -i https://pypi.rbln.ai/simple/ rebel-compiler
  if [ $? -ne 0 ]; then
    echo "An error occured while Rebellions package installation."
    popd
    exit 1
  fi
  pushd ${DIR} && \
    git submodule update --init ./ultralytics && \
	  pip3 install -r ./requirements.txt && \
	  python3 ./compile.py --model_name yolov8l
  if [ $? -ne 0 ]; then
    echo "An error occurred while compiling the model."
    popd
    exit 1
  fi
  popd
  cp -v ${DIR}/${LABEL_SRC}/${LABEL_FILE} ${LABEL_DST}
  if [ $? -ne 0 ]; then
    echo "An error occurred while compiling the model."
    popd
    exit 1
  fi
else
  echo " \"${MODEL_PATH}\" already exists."
  echo " \"${LABEL_DST}/${LABEL_FILE}\" already exists."
fi

echo "=================================================================="
echo "= Step 2. Clone python_backend                                   ="
echo "=================================================================="

BE_DIR="python_backend"
GIT_REPO="https://github.com/triton-inference-server/${BE_DIR}.git"
if [ ! -d ${BE_DIR} ]; then
  git clone ${GIT_REPO} -b r26.07
  if [ $? -ne 0 ]; then
    echo "Error while cloning git repository."
    popd
    exit 1
  fi  
else
  echo " \"${BE_DIR}\" already exists."
fi

echo "=================================================================="
echo "= Step 3. Copy model & python source to repository dir           ="
echo "=================================================================="

MODEL_DIR="python_backend/examples/rbln/${MODEL}/1"

if [ ! -d ${MODEL_DIR} ] || [ ! -f ${MODEL_DIR}/${MODEL}.rbln ]; then
  mkdir -p ${MODEL_DIR}
  if [ ! -d ${MODEL_DIR} ]; then
    echo "Failed to make model directory : \"${MODEL_DIR}\"."
    popd
    exit 1
  fi  
  cp -v ${MODEL_PATH} ${MODEL_DIR}/
  if [ $? -ne 0 ]; then
    echo "Failed to copy model directory to model repository : \"${MODEL_DIR}\"."
    popd
    exit 1
  fi  
  cp -v ../material/model.py ${MODEL_DIR}/
  if [ $? -ne 0 ]; then
    echo "Failed to copy model python to model repository : \"${MODEL_DIR}\"."
    popd
    exit 1
  fi
else
  echo " \"${MODEL_DIR}\" and \"${MODEL_DIR}/${MODEL}.rbln\" already exists."
fi

echo "=================================================================="
echo "= Step 4. Copy config.pbtxt to model repository                  ="
echo "=================================================================="

REPO_DIR="python_backend/examples/rbln/${MODEL}/"
CONFIG="config.pbtxt"

if [ ! -f ${REPO_DIR}/${CONFIG} ]; then
  cp -v ../material/${CONFIG} ${REPO_DIR}/
  if [ $? -ne 0 ]; then
    echo "Failed to copy configuration file : \"${CONFIG}\"."
    popd
    exit 1
  fi
fi

echo "=================================================================="
echo "= Step 5. Install required packages                              ="
echo "=================================================================="
pip3 install -r ../material/requirements.txt

if [ $? -ne 0 ]; then
  echo "Error while required packages installation."
  popd
  exit 1
fi

echo "Done."

popd > /dev/null
exit 0

