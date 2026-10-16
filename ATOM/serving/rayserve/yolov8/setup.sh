#!/bin/bash

OUTPUT_DIR="output"
if [ ! -d ${OUTPUT_DIR} ]; then
  mkdir ${OUTPUT_DIR}
fi

pushd ${OUTPUT_DIR}

echo "=================================================================="
echo "= Step 1. Directory check & preparation                          ="
echo "=================================================================="

MATERIAL_PATH="../material"
SERVE_CODE="serve.py"

if [ ! -f ./${SERVE_CODE} ]; then
  cp -v ${MATERIAL_PATH}/${SERVE_CODE} ./
else
  echo "./${SERVE_CODE} is already exists."
fi

if [ $? -ne 0 ]; then
  echo "Error while directory preparation."
fi

echo "=================================================================="
echo "= Step 2. Install required packages                              ="
echo "=================================================================="
pip3 install -r ${MATERIAL_PATH}/requirements.txt

if [ $? -ne 0 ]; then
  echo "Error while required packages installation."
  popd
  exit 1
fi

echo "=================================================================="
echo "= Step 3. YOLOv8 model compile                                   ="
echo "=================================================================="

DIR="../material/yolov8"
MODEL="yolov8l"
MODEL_PATH="${DIR}/${MODEL}.rbln"
MODEL_DIR="."
LABEL_SRC="ultralytics/ultralytics/cfg/datasets"
LABEL_FILE="coco128.yaml"
if [ ! -f ${MODEL_DIR}/${MODEL}.rbln ] || [ ! -f ${MODEL_DIR}/${LABEL_FILE} ]; then
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
  cp -v ${MODEL_PATH} ${MODEL_DIR}
  cp -v ${DIR}/${LABEL_SRC}/${LABEL_FILE} ${MODEL_DIR}/
  if [ $? -ne 0 ]; then
    echo "An error occurred while compiling the model."
    popd
    exit 1
  fi
else
  echo " \"${MODEL_DIR}/${MODEL}.rbln\" already exists."
  echo " \"${MODEL_DIR}/${LABEL_FILE}\" already exists."
fi

echo "Done."

echo "All preparation is done."
echo "You can start with serve cli in \"output\" directory(ex. \"serve run serve:app --name yolov8\")."
echo "Example request:"
echo "    $ curl -X POST http://localhost:8000/ \\"
echo "        --header \"Content-Type: image/jpeg\" \\"
echo "        --data-binary @<PATH>/<IMAGE_FILE> | jq ."

popd > /dev/null
