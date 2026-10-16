#!/bin/bash

OUTPUT_DIR="output"

if [ ! -d ${OUTPUT_DIR} ]; then
  mkdir ${OUTPUT_DIR}
fi

pushd ${OUTPUT_DIR}

echo "=================================================================="
echo "= Step 1. Resnet50 model compile                                 ="
echo "=================================================================="

DIR="resnet-50"
MODEL="resnet50"
if [ ! -d ${DIR} ]; then
  python3 ../material/compile.py
  if [ $? -ne 0 ];then
    echo "An error occurred while compiling the model."
    popd
    exit 1
  fi  
  if [ ! -f ./${MODEL}.rbln ]; then
    echo "Error. the compiled \"${MODEL}.rbln\" file is missing."
    popd
    exit 1
  fi  
else
  echo " \"${DIR}\" already exists."
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

if [ ! -d ${MODEL_DIR} ]; then
  mkdir -p ${MODEL_DIR}
  if [ ! -d ${MODEL_DIR} ]; then
    echo "Failed to make model directory : \"${MODEL_DIR}\"."
    popd
    exit 1
  fi  
  cp -v ./${MODEL}.rbln ${MODEL_DIR}/ 
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
  echo " \"${MODEL_DIR}\" already exists."
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

echo "Done."

popd > /dev/null

