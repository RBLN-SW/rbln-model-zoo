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
                      "optimum-rbln"
                      "torchserve"
                      "torch-model-archiver"
                      "torch-workflow-archiver")

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

function check_torchserve() {
  if command -v torchserve >/dev/null 2>&1; then
    return 0
  else
    return 1
  fi  
}

function check_torch-model-archiver() {
  if command -v torch-model-archiver >/dev/null 2>&1; then
    return 0
  else
    return 1
  fi  
}

BASE_DIR="${PWD}/output/model_store/llama3.1-8b"
CONFIG_DIR="${PWD}/output/model_config/model_config.yaml"
TARGET_MODEL_DIR="Llama-3.1-8B-Instruct"

function check_model_infos() {
  echo "model informations check(no archive format)"
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
  if [ -f "${CONFIG_DIR}" ] ; then
    echo " * ${CONFIG_DIR} : ${colored_OK}"
  else
    echo " * ${CONFIG_DIR} : ${colored_NO}"
    return_val=1
  fi

  return ${return_val}
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

if check_torchserve; then
  echo "torchserve check : ${colored_OK}"
else
  echo "torchserve check : ${colored_NO}"
  echo " * torchserve is not installed."
  echo " * Please refer to TorchServe \"Getting Start\" document(https://pytorch.org/serve/getting_started.html](https://pytorch.org/serve/getting_started.html)."
fi

if check_torch-model-archiver; then
  echo "torch-model-archiver check : ${colored_OK}"
else
  echo "torch-model-archiver check : ${colored_NO}"
  echo " * torch-model-archiver is not installed."
  echo " * Please refer to TorchServe \"Getting Start\" document(https://pytorch.org/serve/getting_started.html](https://pytorch.org/serve/getting_started.html)."
fi

if check_model_infos; then
  echo "model-store directory check : ${colored_OK}"
else
  echo "model-store directory check : ${colored_NO}"
  echo " * Please refer to Rebellions document(https://doc.rbln.ai)."
fi

