#!/bin/bash

###############
# Color print #
###############
RED=$(tput setaf 1)     # Red color
GREEN=$(tput setaf 2)   # Green color
RESET=$(tput sgr0)      # Reset color
colored_NO="[${RED}NO${RESET}]"
colored_OK="[${GREEN}OK${RESET}]"

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

MODEL_ARCHIVE="${PWD}/output/model_store/yolov8l.mar"
CONFIG_DIR="${PWD}/output/config.properties"

function check_model_infos() {
  echo "model informations check(archive format)"
  return_val=0
  if [ -f "${MODEL_ARCHIVE}" ]; then
    echo " * ${MODEL_ARCHIVE} : ${colored_OK}"
  else
    echo " * ${MODEL_ARCHIVE} : ${colored_NO}"
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

