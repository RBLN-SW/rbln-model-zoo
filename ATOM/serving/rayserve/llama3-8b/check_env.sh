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
                      "vllm_rbln"
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

TARGET_MODEL_DIR="Meta-Llama-3-8B-Instruct"

function check_model_infos() {
  echo "model informations check"
  return_val=0
  if [ -d "output/${TARGET_MODEL_DIR}" ]; then
    echo " * output/${TARGET_MODEL_DIR} : ${colored_OK}"
  else
    echo " * output/${TARGET_MODEL_DIR} : ${colored_NO}"
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
  echo "Required packages check : ${colored_OK}"
else
  echo "Required packages check : ${colored_NO}"
fi

if check_ray_serve; then
  echo "ray serve check : ${colored_OK}"
else
  echo "ray serve check : ${colored_NO}"
  echo " * ray serve is not installed."
  echo " * Please refer to Ray Serve \"Getting Start\" document(https://docs.ray.io/en/latest/serve/getting_started.html)."
fi

if check_model_infos; then
  echo "model directory check : ${colored_OK}"
else
  echo "model directory check : ${colored_NO}"
  echo " * Please refer to Rebellions document(https://docs.rbln.ai)."
fi

