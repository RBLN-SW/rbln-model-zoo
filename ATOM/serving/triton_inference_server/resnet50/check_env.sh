#!/bin/bash

###############
# Color print #
###############
RED=$(tput setaf 1)     # Red color
GREEN=$(tput setaf 2)   # Green color
RESET=$(tput sgr0)      # Reset color
colored_NO="[${RED}NO${RESET}]"
colored_OK="[${GREEN}OK${RESET}]"

function check_huggingface_login() {
  output=$(huggingface-cli whoami 2>&1)

  if [[ "$output" == *"Not logged in"* ]]; then
    return 1
  else
    return 0
  fi  
}

required_packages=("rebel-compiler"
                      "optimum-rbln")

function check_packages_installed() {
  echo "Required packages installed check"
  all_installed=true

  for package in "${required_packages[@]}"; do
    if ! pip list --format=freeze --disable-pip-version-check 2>/dev/null | grep -q "^${package}=="; then
      echo " * '${package}' package check : ${colored_NO}"
      all_installed=false
    else
      echo " * '${package}' package check : ${colored_OK}"
    fi  
  done

  if $all_installed; then
    return 0
  else
    return 1
  fi  
}

function check_triton_server() {
  if command -v tritonserver >/dev/null 2>&1; then
    return 0
  else
    return 1
  fi  
}

function check_python_backend() {
  if [ -d "./output/python_backend" ]; then
    return 0
  else
    return 1
  fi  
}

BASE_DIR="${PWD}/output/python_backend/examples/rbln"
TARGET_MODEL_DIR="resnet50"
MODEL="${TARGET_MODEL_DIR}.rbln"
TARGET_MODEL_JSON="model.json"

function check_model_infos() {
  echo "model informations check"
  return_val=0
  if [ -d "${BASE_DIR}" ]; then
    echo " * ${BASE_DIR} : ${colored_OK}"
  else
    echo " * ${BASE_DIR} : ${colored_NO}"
    return_val=1
  fi
  if [ -d "${BASE_DIR}/${TARGET_MODEL_DIR}" ]; then
    echo " * ${BASE_DIR}/${TARGET_MODEL_DIR} : ${colored_OK}"
  else
    echo " * ${BASE_DIR}/${TARGET_MODEL_DIR} : ${colored_NO}"
    return_val=1
  fi
  if [ -f "${BASE_DIR}/${TARGET_MODEL_DIR}/config.pbtxt" ] ; then
    echo " * ${BASE_DIR}/${TARGET_MODEL_DIR}/config.pbtxt : ${colored_OK}"
  else
    echo " * ${BASE_DIR}/${TARGET_MODEL_DIR}/config.pbtxt : ${colored_NO}"
    return_val=1
  fi
  if [ -f "${BASE_DIR}/${TARGET_MODEL_DIR}/1/model.py" ] ; then
    echo " * ${BASE_DIR}/${TARGET_MODEL_DIR}/1/model.py : ${colored_OK}"
  else
    echo " * ${BASE_DIR}/${TARGET_MODEL_DIR}/1/model.py : ${colored_NO}"
    return_val=1
  fi
  if [ -f "${BASE_DIR}/${TARGET_MODEL_DIR}/1/${MODEL}" ] ; then
    echo " * ${BASE_DIR}/${TARGET_MODEL_DIR}/1/${MODEL} : ${colored_OK}"
  else
    echo " * ${BASE_DIR}/${TARGET_MODEL_DIR}/1/${MODEL} : ${colored_NO}"
    return_val=1
  fi

  return ${return_val}
}

function check_model_path() {
  if [ ! -f ${BASE_DIR}/${TARGET_MODEL_JSON} ]; then
    return 1
  fi
  model_val=`grep '"model":' ${BASE_DIR}/${TARGET_MODEL_JSON} | sed -E 's/.*"model": "([^"]+)",.*/\1/'`
  if [ ! -d ${model_val} ]; then
    return 1
  fi
  return 0
}

if check_huggingface_login; then
  echo "Huggingface login check : ${colored_OK}"
else
  echo "Huggingface login check : ${colored_NO}"
  echo " * Huggingface login is needed."
  echo " * Please refer to \"https://huggingface.co/docs/huggingface_hub/quick-start#login-command\"."
fi


if check_packages_installed; then
  echo "Package check : ${colored_OK}"
else
  echo "Package check : ${colored_NO}"
fi

if check_triton_server; then
  echo "Triton Server check : ${colored_OK}"
else
  echo "Triton Server check : ${colored_NO}"
  echo " * Triton server is not installed."
  echo " * Please refer to Rebellions document(https://doc.rbln.ai)."
fi

if check_python_backend; then
  echo "python backend check : ${colored_OK}"
else
  echo "python backend check : ${colored_NO}"
  echo " * Please clone triton inference server python backend."
  echo " * ex) $ git clone https://github.com/triton-inference-server/python_backend -b r24.01"
  echo " * Please refer to Rebellions document(https://doc.rbln.ai)."
fi

if check_model_infos; then
  echo "Model repository check : ${colored_OK}"
else
  echo "Model repository check : ${colored_NO}"
  echo "Model repository validation check failed."
  echo " * Please refer to Rebellions document(https://doc.rbln.ai)."
fi

