#!/bin/bash

echo "======================================"
echo "== Go1 Sim-to-Real Installation Kit =="
echo "======================================"
echo ""
echo "Author: Gabriel Margolis, Improbable AI Lab, MIT"
echo "This software is intended to support controls research. It includes safety features but may still damage your Go1. The user assumes all risk."
echo ""

read -r -p "Extract docker container? [y/N] " response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]
then
    # load docker image
    echo "[Step 1] Extracting docker image..."
    docker load -i ../scripts/deployment_image.tar
    printf "\nDone!\n"
else
    echo "Quitting"
fi


