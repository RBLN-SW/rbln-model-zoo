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
  "rbln_models/yolov8l"
  "rbln_extra"
  "model_handler"
  "model_store"
)

for dir in "${directories[@]}"; do
    if [ ! -d "${dir}" ]; then
      mkdir -p "${dir}"
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
echo "= Step 2. Install required packages                              ="
echo "=================================================================="
pip3 install -r ${MATERIAL_PATH}/requirements.txt

if [ $? -ne 0 ]; then
  echo "Error while required packages installation."
  popd
  exit 1
fi

echo "=================================================================="
echo "= Step 3. YOLOv8 model compile                                ="
echo "=================================================================="

DIR="../material/yolov8"
MODEL="yolov8l"
MODEL_PATH="${DIR}/${MODEL}.rbln"
MODEL_DST="./rbln_models/${MODEL}"
LABEL_DST="./rbln_extra"
LABEL_SRC="ultralytics/ultralytics/cfg/datasets"
LABEL_FILE="coco128.yaml"
if [ ! -f ${MODEL_DST}/${MODEL}.rbln ] || [ ! -f ${LABEL_DST}/${LABEL_FILE} ]; then
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
  cp -v ${MODEL_PATH} ${MODEL_DST}
  cp -v ${DIR}/${LABEL_SRC}/${LABEL_FILE} ${LABEL_DST}/
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
echo "= Step 3. Model archiving(archive format)                     ="
echo "=================================================================="

SERVING_MODEL_NAME=${MODEL}
ARCHIVE_RESULT_PATH="./model_store/${SERVING_MODEL_NAME}.mar"

if [ ! -f ${ARCHIVE_RESULT_PATH} ]; then
  torch-model-archiver \
    --model-name ${SERVING_MODEL_NAME} \
    --version 1.0 \
    --serialized-file ./rbln_models/${MODEL}/${MODEL}.rbln \
    --handler ./model_handler/rbln_handler.py \
    --extra-files ./rbln_extra/coco128.yaml \
    --export-path ./model_store

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
