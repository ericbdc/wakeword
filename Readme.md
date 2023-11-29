# Wakeword

Allows to train a personalised wakeword model thanks to [OpenWakeWord](https://github.com/dscripka/openWakeWord/). 

It modifies a bit the [provided great python notebook](https://github.com/dscripka/openWakeWord/blob/main/notebooks/automatic_model_training.ipynb) so that you can use it directly as a [`automatic_model_training.py`](./scripts/automatic_model_training.py) file (then you can use it without any IPython kernel).

## Troubleshoots

For now it is not available on MacOS (because of [piper-phonemize](https://github.com/rhasspy/piper-phonemize) library), then you will have to train it on a Linux platform.

## Launch on a remote cloud machine

A set of scripts are available in this repository to rapidly configure a remote machine.

### Clone this repository on your local machine

```zsh
git clone git@github.com:ericbdc/wakeword.git
cd wakeword
```

### Get a remote machine

First you need a running Linux platform available, e.g.: AWS EC2 Ubuntu (all runs in ~30min on a t3.2xlarge).

:warning: Beware that you will need about 32Gb space disk if your chose only 1 hour clips.

In that case, you will generate a Key Pair, name it `wakeword` and download the `wakeword.pem` file to your `~/.ssh/` folder.

Check you can connect through ssh, replace `user@ip` with yours:
```zsh
ssh -i ~/.ssh/wakeword.pem remote_user@ec2-xx-xx-xx-xxx.eu-west-3.compute.amazonaws.com
```

### Configure remote machine

Copy useful scripts
```zsh
./send_to_remote.sh 'ec2-xx-xx-xx-xxx.eu-west-3.compute.amazonaws.com'
```

Do the basic setup on the remote machine
```zsh
ssh -i ~/.ssh/wakeword.pem remote_user@ec2-xx-xx-xx-xxx.eu-west-3.compute.amazonaws.com ./setup_remote.sh
```

### Train your model with your wake sentence

Better launch the script in a Linux screen
```zsh
screen -S wakeword
```

```zsh
ssh -i ~/.ssh/wakeword.pem remote_user@ec2-xx-xx-xx-xxx.eu-west-3.compute.amazonaws.com python automatic_model_training.py | at now -m
```

You can detach your screen session by doing Ctrl+A Ctrl+D.
If you want to re-attach your screen do ```screen -R wakeword```.
To exit your screen do ```exit``` and press Enter. You will lose the shell outputs as screen launchs a subshell.

## Regenerating the `automatic_model_training.py` file

Activate the venv and install dev requirements locally
```zsh
python -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt
```

```zsh
jupyter nbconvert automatic_model_training.ipynb --to python
```

If you made any change, you will need to copy it again to your remote machine.

## Testing the wakeword model

```zsh
python detect_from_microphone.py --inference_framework=onnx --model_path='my_custom_model/simbad.onnx'
```
