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
                      "ray")

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

function check_ray_serve() {
  if python3 -c "from ray import serve" >/dev/null 2>&1; then
    return 0
  else
    return 1
  fi  
}

MODEL_FILE="${PWD}/output/resnet50.rbln"

function check_model_infos() {
  echo "model file check"
  return_val=0
  if [ -f "${MODEL_FILE}" ]; then
    echo " * ${MODEL_FILE} : ${colored_OK}"
  else
    echo " * ${MODEL_FILE} : ${colored_NO}"
    return_val=1
  fi

  return ${return_val}
}

if check_packages_installed; then
  echo "Ray Package check : ${colored_OK}"
else
  echo "Ray Package check : ${colored_NO}"
fi

if check_ray_serve; then
  echo "ray serve check : ${colored_OK}"
else
  echo "ray serve check : ${colored_NO}"
  echo " * ray serve is not installed."
  echo " * Please refer to Ray Serve \"Getting Start\" document(https://docs.ray.io/en/latest/serve/getting_started.html)."
fi

if check_model_infos; then
  echo "model file check : ${colored_OK}"
else
  echo "model file check : ${colored_NO}"
  echo " * Please refer to Rebellions document(https://docs.rbln.ai)."
fi

