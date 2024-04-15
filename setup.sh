#!/bin/bash
# Creates a virtual environment and installs the required packages.
python3 -m venv image-env
source image-env/bin/activate
pip install -r requirements.txt
deactivate
