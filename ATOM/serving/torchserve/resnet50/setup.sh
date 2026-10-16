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
MODEL_CONFIG="config.properties"
MODEL_HANDLER="rbln_handler.py"

directories=(
  "rbln_models"
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

if [ ! -f ./${MODEL_CONFIG} ]; then
  cp -v ${MATERIAL_PATH}/${MODEL_CONFIG} ./
else
  echo "./${MODEL_CONFIG} is already exists."
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
echo "= Step 2. Resnet50 model compile                                ="
echo "=================================================================="

MODEL_NAME="resnet50"
MODEL_FILE="${MODEL_NAME}.rbln"
MODEL_DIR="rbln_models"

if [ ! -d ${MODEL_DIR}/${MODEL_NAME} ] || [ ! -f ${MODEL_DIR}/${MODEL_NAME}/${MODEL_FILE} ]; then
  mkdir -p ${MODEL_DIR}/${MODEL_NAME}
  pushd ${MATERIAL_PATH} && 
    python ./compile.py --model_name resnet50 &&
    mv ${MODEL_FILE} ../output/${MODEL_DIR}/${MODEL_NAME}/${MODEL_FILE} && 
    popd

  if [ $? -ne 0 ] || [ ! -f ${MODEL_DIR}/${MODEL_NAME}/${MODEL_FILE} ]; then
    echo "Error while model compile."
    popd
    exit 1
  fi
else
  echo " \"${MODEL_DIR}/${MODEL_NAME}/${MODEL_FILE}\" already exists."
fi

echo "=================================================================="
echo "= Step 3. Model archiving(archive format)                     ="
echo "=================================================================="

SERVING_MODEL_NAME="resnet50"
ARCHIVE_RESULT_PATH="./model_store/${SERVING_MODEL_NAME}.mar"

if [ ! -f ${ARCHIVE_RESULT_PATH} ]; then
  torch-model-archiver \
    --model-name ${SERVING_MODEL_NAME} \
    --version 1.0 \
    --serialized-file ./rbln_models/${MODEL_NAME}/${MODEL_FILE} \
    --handler ./model_handler/rbln_handler.py \
    --export-path ./model_store/

  if [ ! -f ${ARCHIVE_RESULT_PATH} ]; then
    echo "Error while model archiving."
    popd
    exit 1
  fi
else
  echo "\"${ARCHIVE_RESULT_PATH}\" aleady exists."
fi

echo "Done."

popd > /dev/null
