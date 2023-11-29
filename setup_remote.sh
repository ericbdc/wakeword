#!/bin/bash

sudo apt update -y && sudo apt upgrade -y
sudo add-apt-repository ppa:fkrull/deadsnakes -y # ppa:deadsnakes/ppa
sudo apt update -y
sudo apt install -y python3.10
alias python='python3.10'
sudo apt install -y python3.10-dev # necessary for webrtcvad
sudo apt install -y python3.10-venv \
                    python3.10-distutils \
                    python3.10-lib2to3 \
                    python3-pip \
                    python-is-python3
export PATH=$PATH:/home/ubuntu/.local/bin
sudo apt install at
sudo python3 -m pip install --upgrade pip
python -m venv venv
source venv/bin/activate
