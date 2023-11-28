#!/bin/bash

for file in 'setup_remote.sh' 'automatic_model_training.py'
do
    scp -i ~/.ssh/wakeword.pem $file ubuntu@$1:$file
done
