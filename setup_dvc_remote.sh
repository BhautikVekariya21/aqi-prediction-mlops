#!/bin/bash
echo "Setting up DagsHub remote..."
dvc remote add --default dagshub https://dagshub.com/${DAGSHUB_REPO_OWNER}/${DAGSHUB_REPO_NAME}.dvc
dvc remote modify dagshub --local auth basic
dvc remote modify dagshub --local user ${DAGSHUB_REPO_OWNER}
dvc remote modify dagshub --local password ${DAGSHUB_TOKEN}
echo "DagsHub remote configured!"