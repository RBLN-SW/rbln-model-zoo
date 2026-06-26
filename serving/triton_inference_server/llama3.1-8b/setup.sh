#!/bin/bash
OUTPUT_DIR="output"
if [ ! -d ${OUTPUT_DIR} ]; then
  mkdir ${OUTPUT_DIR}
fi

pushd ${OUTPUT_DIR}

echo "=================================================================="
echo "= Step 1. Clone vllm_backend                                     ="
echo "=================================================================="

DIR="vllm_backend"
GIT_REPO="https://github.com/triton-inference-server/${DIR}.git"
if [ ! -d ${DIR} ]; then
  git clone ${GIT_REPO} -b r26.06
  if [ $? -ne 0 ]; then
    echo "Error while cloning git repository."
    popd
    exit 1
  fi  
else
  echo " \"${DIR}\" already exists."
fi

echo "=================================================================="
echo "= Step 2. Llama3.1-8B model compile                                ="
echo "=================================================================="

DIR="Llama-3.1-8B-Instruct"
if [ ! -d ${DIR} ]; then
  python3 ../material/compile.py
  if [ $? -ne 0 ];then
    echo "An error occurred while compiling the model."
    popd
    exit 1
  fi  
else
  echo " \"${DIR}\" already exists."
fi

echo "=================================================================="
echo "= Step 3. Copy model dir to vllm_model dir                       ="
echo "=================================================================="

MODEL_DIR="vllm_backend/samples/model_repository/vllm_model/1"
DIR="Llama-3.1-8B-Instruct"

if [ ! -d ${MODEL_DIR}/${DIR} ]; then
  cp -rfv ./${DIR} ./vllm_backend/samples/model_repository/vllm_model/1
  if [ $? -ne 0 ]; then
    echo "Error while copying model directory to vllm_model path."
    popd
    exit 1
  fi  
else
  echo " \"${MODEL_DIR}/${DIR}\" already exists."
fi

echo "=================================================================="
echo "= Step 4. Apply model absolute path to config.json               ="
echo "=================================================================="

DIR="meta-llama/Llama-3.1-8B-Instruct"
MODEL_PATH="vllm_backend/samples/model_repository/vllm_model"
MODEL_DIR="${MODEL_PATH}/1"
CONFIG="model.json"
ABSOLUTE_PATH="${PWD}/${MODEL_DIR}"

JSON_CONTENT=$(printf '{
    "model": "%s/Llama-3.1-8B-Instruct",
    "max_num_seqs": 1,
    "max_num_batched_tokens": 131072,
    "max_model_len": 131072,
    "block_size": 16384,
    "device": "rbln"
}' "${ABSOLUTE_PATH}")

if [ -f ${MODEL_DIR}/${CONFIG} ]; then
  echo ${JSON_CONTENT} | jq . > ${MODEL_DIR}/${CONFIG}
  if [ $? -ne 0 ]; then
    echo "Failed to configuration file : \"${CONFIG}\"."
    popd
    exit 1
  fi
  echo "Done"
else
  echo "Error. \"${MODEL_DIR}/${CONFIG}\" file does not exist."
  popd
  exit 1
fi

popd > /dev/null
