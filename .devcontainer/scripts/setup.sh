#!/bin/bash

# @description Ensures the Kaggle configuration directory exists.
#              Creates ~/.config/kaggle if it does not already exist.
ensure_kaggle_config_dir() {
  local kaggle_config_dir="${HOME}/.config/kaggle"

  if [ ! -d "${kaggle_config_dir}" ]; then
    echo "Creating Kaggle config directory: ${kaggle_config_dir}"
    mkdir -p "${kaggle_config_dir}"
  fi
}

# To use the function, you can call it like this:
# ensure_kaggle_config_dir