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
echo "= Step 3. Llama3.1-8B model compile                              ="
echo "=================================================================="

LLAMA_3_1B_DIR="Llama-3.1-8B-Instruct"
MODEL_DIR="."

if [ ! -d ${MODEL_DIR}/${LLAMA_3_1B_DIR} ]; then
  pushd ${MATERIAL_PATH} && 
    python ./compile.py &&
    mv ${LLAMA_3_1B_DIR} ../output/${MODEL_DIR}/ && 
    popd

  if [ $? -ne 0 ] || [ ! -d ${MODEL_DIR}/${LLAMA_3_1B_DIR} ]; then
    echo "Error while model compile."
    popd
    exit 1
  fi
else
  echo " \"${MODEL_DIR}/${LLAMA_3_1B_DIR}\" already exists."
fi


echo "Done."

echo "All preparation is done."
echo "You can start with serve cli in \"output\" directory(ex. \"serve run serve:app --name llama3.1-8b\")."

echo "Example request:"
echo "$ curl -sN -H \"Content-Type: application/json\" -X POST http://localhost:8000/v1/completions \\"
echo "    --data-raw '{"
echo "      \"model\": \"Llama-3.1-8B-Instruct\","
echo "      \"prompt\": \"A robot may not injure a human being\","
echo "      \"stream\": false"
echo "    }' | jq ."

popd > /dev/null
