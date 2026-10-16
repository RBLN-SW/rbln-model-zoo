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
MODEL_CONFIG="model_config.yaml"
MODEL_HANDLER="rbln_vllm_handler.py"

directories=(
  "rbln_model"
  "model_config"
  "model_handler"
  "model_store"
)

for dir in "${directories[@]}"; do
    if [ ! -d "${dir}" ]; then
      mkdir "${dir}"
    else
      echo  "${dir} is already exists."
    fi
done

if [ ! -f ./model_config/${MODEL_CONFIG} ]; then
  cp -v ${MATERIAL_PATH}/${MODEL_CONFIG} ./model_config/
else
  echo "./model_config/${MODEL_CONFIG} is already exists."
fi

if [ ! -f ./model_handler/${MODEL_HANDLER} ]; then
  cp -v ${MATERIAL_PATH}/${MODEL_HANDLER} ./model_handler/
else
  echo "./model_handler/${MODEL_HANDLER} is already exists."
fi

if [ $? -ne 0 ]; then
  echo "Error while directory preparation."
fi

echo "=================================================================="
echo "= Step 2. Llama3.1-8B model compile                              ="
echo "=================================================================="

LLAMA_3_1B_DIR="Llama-3.1-8B-Instruct"
MODEL_DIR="rbln_model"

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

echo "=================================================================="
echo "= Step 3. Model archiving(no-archive format)                     ="
echo "=================================================================="

SERVING_MODEL_NAME="llama3.1-8b"
ARCHIVE_RESULT_PATH="./model_store/${SERVING_MODEL_NAME}"

if [ ! -d ${ARCHIVE_RESULT_PATH} ]; then
  torch-model-archiver --model-name ${SERVING_MODEL_NAME} \
    --version 1.0 --handler vllm_handler \
    --config-file ./model_config/model_config.yaml \
    --handler ./model_handler/rbln_vllm_handler.py \
    --archive-format no-archive \
    --export-path model_store/ \
    --extra-files rbln_model/

  if [ ! -d ${ARCHIVE_RESULT_PATH} ]; then
    echo "Error while model archiving."
    popd
    exit 1
  fi
else
  echo "\"${ARCHIVE_RESULT_PATH}\" aleady exists."
fi

echo "Done."

popd > /dev/null
