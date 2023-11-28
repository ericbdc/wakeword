#!/bin/bash

sudo apt update
sudo apt install at
sudo apt install -y python3-pip
export PATH=$PATH:/home/ubuntu/.local/bin
sudo apt-get install -y python3.10-venv
sudo apt install -y python-is-python3
python -m pip install --upgrade pip
python -m venv venv
source venv/bin/activate
