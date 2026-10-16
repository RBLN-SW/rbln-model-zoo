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
  echo "Error while setup preparation."
  return 1
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
echo "= Step 3. Resnet50 model compile                                 ="
echo "=================================================================="

MODEL_NAME="resnet50"
MODEL_FILE="${MODEL_NAME}.rbln"
MODEL_DIR="."

if [ ! -f ${MODEL_DIR}/${MODEL_FILE} ]; then
  pushd ${MATERIAL_PATH} && 
    python ./compile.py --model_name resnet50 && 
    mv ${MODEL_FILE} ../output/${MODEL_DIR} && 
    popd

  if [ $? -ne 0 ] || [ ! -f ${MODEL_DIR}/${MODEL_FILE} ]; then
    echo "Error while model compile."
    popd
    exit 1
  fi
else
  echo " \"${MODEL_DIR}/${MODEL_FILE}\" already exists."
fi

echo "Done."

echo "All preparation is done."
echo "You can start with serve cli in \"output\" directory(ex. \"serve run serve:app --name resnet50\")."
echo "Example request:"
echo "    $ curl -X POST http://localhost:8000/ \\"
echo "        --header \"Content-Type: image/jpeg\" \\"
echo "        --data-binary @<PATH>/<IMAGE_FILE> | jq ."

popd > /dev/null
