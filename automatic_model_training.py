#!/usr/bin/env python
# coding: utf-8

# # Introduction

# This notebook demonstrates how to train custom openWakeWord models using pre-defined datasets and an automated process for dataset generation and training. While not guaranteed to always produce the best performing model, the methods shown in this notebook often produce baseline models with releatively strong performance.
# 
# Manual data preparation and model training (e.g., see the [training models](training_models.ipynb) notebook) remains an option for when full control over the model development process is needed.
# 
# At a high level, the automatic training process takes advantages of several techniques to try and produce a good model, including:
# 
# - Early-stopping and checkpoint averaging (similar to [stochastic weight averaging](https://arxiv.org/abs/1803.05407)) to search for the best models found during training, according to the validation data
# - Variable learning rates with cosine decay and multiple cycles
# - Adaptive batch construction to focus on only high-loss examples when the model begins to converge, combined with gradient accumulation to ensure that batch sizes are still large enough for stable training
# - Cycical weight schedules for negative examples to help the model reduce false-positive rates
# 
# See the contents of the `train.py` file for more details.

# # Environment Setup

# To begin, we'll need to install the requirements for training custom models. In particular, a relatively recent version of Pytorch and custom fork of the [piper-sample-generator](https://github.com/dscripka/piper-sample-generator) library for generating synthetic examples for the custom model.
# 
# **Important Note!** Currently, automated model training is only supported on linux systems due to the requirements of the text to speech library used for synthetic sample generation (Piper). It may be possible to use Piper on Windows/Mac systems, but that has not (yet) been tested.

# In[ ]:


# ## Environment setup

# # install piper-sample-generator (currently only supports linux systems)
# !git clone https://github.com/rhasspy/piper-sample-generator
# !wget -O piper-sample-generator/models/en_US-libritts_r-medium.pt 'https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/en_US-libritts_r-medium.pt'
# # !pip install pybind11==2.11.1 
# !pip install piper-phonemize
# !pip install webrtcvad

# # install openwakeword (full installation to support training)
# !git clone https://github.com/dscripka/openwakeword
# !pip install -e ./openwakeword
# !cd openwakeword

# # install other dependencies
# !pip install mutagen==1.47.0
# !pip install torchinfo==1.8.0
# !pip install torchmetrics==1.2.0
# !pip install speechbrain==0.5.14
# !pip install audiomentations==0.33.0
# !pip install torch-audiomentations==0.11.0
# !pip install acoustics==0.2.6
# !pip install tensorflow-cpu==2.8.1
# !pip install tensorflow_probability==0.16.0
# !pip install onnx_tf==1.10.0
# !pip install pronouncing==0.2.0
# !pip install datasets==2.14.6
# !pip install deep-phonemizer==0.0.19

# # Download required models (workaround for Colab)
# import os
# os.makedirs("./openwakeword/openwakeword/resources/models", exist_ok=True)
# !wget https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.onnx -O ./openwakeword/openwakeword/resources/models/embedding_model.onnx
# !wget https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.tflite -O ./openwakeword/openwakeword/resources/models/embedding_model.tflite
# !wget https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.onnx -O ./openwakeword/openwakeword/resources/models/melspectrogram.onnx
# !wget https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.tflite -O ./openwakeword/openwakeword/resources/models/melspectrogram.tflite


# In[ ]:


## Environment setup
import os

os.system('source venv/bin/activate')

# install piper-sample-generator (currently only supports linux systems)
os.system('git clone https://github.com/rhasspy/piper-sample-generator')
os.system("wget -O piper-sample-generator/models/en_US-libritts_r-medium.pt 'https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/en_US-libritts_r-medium.pt'")
os.system('pip install piper-phonemize')
os.system('pip install webrtcvad')
os.system('pip install -r piper-sample-generator/requirements.txt')

# install openwakeword (full installation to support training)
os.system('git clone https://github.com/dscripka/openwakeword')
os.system('pip install -e ./openwakeword')
os.system('cd openwakeword')

# install other dependencies
os.system('pip install mutagen==1.47.0')
os.system('pip install torchinfo==1.8.0')
os.system('pip install torchmetrics==1.2.0')
os.system('pip install speechbrain==0.5.14')
os.system('pip install audiomentations==0.33.0')
os.system('pip install torch-audiomentations==0.11.0')
os.system('pip install acoustics==0.2.6')
os.system('pip install tensorflow-cpu==2.8.1')
os.system('pip install tensorflow_probability==0.16.0')
os.system('pip install onnx_tf==1.10.0')
os.system('pip install pronouncing==0.2.0')
os.system('pip install datasets==2.14.6')
os.system('pip install deep-phonemizer==0.0.19')

# Download required models (workaround for Colab)
os.makedirs("./openwakeword/openwakeword/resources/models", exist_ok=True)
os.system('wget https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.onnx -O ./openwakeword/openwakeword/resources/models/embedding_model.onnx')
os.system('wget https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.tflite -O ./openwakeword/openwakeword/resources/models/embedding_model.tflite')
os.system('wget https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.onnx -O ./openwakeword/openwakeword/resources/models/melspectrogram.onnx')
os.system('wget https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.tflite -O ./openwakeword/openwakeword/resources/models/melspectrogram.tflite')

print("Setup done")


# In[ ]:


# Imports

import os
import numpy as np
import torch
import sys
from pathlib import Path
import uuid
import yaml
import datasets
import scipy
from tqdm import tqdm


# # Download Data

# When training new openWakeWord models using the automated procedure, four specific types of data are required:
# 
# 1) Synthetic examples of the target word/phrase generated with text-to-speech models
# 
# 2) Synthetic examples of adversarial words/phrases generated with text-to-speech models
# 
# 3) Room impulse reponses and noise/background audio data to augment the synthetic examples and make them more realistic
# 
# 4) Generic "negative" audio data that is very unlikely to contain examples of the target word/phrase in the context where the model should detect it. This data can be the original audio data, or precomputed openWakeWord features ready for model training.
# 
# 5) Validation data to use for early-stopping when training the model.
# 
# For the purposes of this notebook, all five of these sources will either be generated manually or can be obtained from HuggingFace thanks to their excellent `datasets` library and extremely generous hosting policy. Also note that while only a portion of some datasets are downloaded, for the best possible performance it is recommended to download the entire dataset and keep a local copy for future training runs.

# In[ ]:


print("Download room impulse responses collected by MIT")
# https://mcdermottlab.mit.edu/Reverb/IR_Survey.html

output_dir = "./mit_rirs"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
rir_dataset = datasets.load_dataset("davidscripka/MIT_environmental_impulse_responses", split="train", streaming=True)

print("Save clips to 16-bit PCM wav files")
for row in tqdm(rir_dataset):
    name = row['audio']['path'].split('/')[-1]
    scipy.io.wavfile.write(os.path.join(output_dir, name), 16000, (row['audio']['array']*32767).astype(np.int16))


# In[ ]:


print("Download noise and background audio")

# Audioset Dataset (https://research.google.com/audioset/dataset/index.html)
# Download one part of the audioset .tar files, extract, and convert to 16khz
# For full-scale training, it's recommended to download the entire dataset from
# https://huggingface.co/datasets/agkphysics/AudioSet, and
# even potentially combine it with other background noise datasets (e.g., FSD50k, Freesound, etc.)
print("Start")
if not os.path.exists("audioset"):
    os.mkdir("audioset")

fname = "bal_train09.tar"
out_dir = f"audioset/{fname}"
link = "https://huggingface.co/datasets/agkphysics/AudioSet/resolve/main/" + fname
os.system(f'wget -O {out_dir} {link}')
os.system('cd audioset && tar -xvf bal_train09.tar')

output_dir = "./audioset_16k"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# Convert audioset files to 16khz sample rate
audioset_dataset = datasets.Dataset.from_dict({"audio": [str(i) for i in Path("audioset/audio").glob("**/*.flac")]})
audioset_dataset = audioset_dataset.cast_column("audio", datasets.Audio(sampling_rate=16000))
for row in tqdm(audioset_dataset):
    name = row['audio']['path'].split('/')[-1].replace(".flac", ".wav")
    scipy.io.wavfile.write(os.path.join(output_dir, name), 16000, (row['audio']['array']*32767).astype(np.int16))

# Free Music Archive dataset (https://github.com/mdeff/fma)
output_dir = "./fma"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
fma_dataset = datasets.load_dataset("rudraml/fma", name="small", split="train", streaming=True)
fma_dataset = iter(fma_dataset.cast_column("audio", datasets.Audio(sampling_rate=16000)))

print("Download clips")
n_hours = 1  # use only 1 hour of clips for this example notebook, recommend increasing for full-scale training
for i in tqdm(range(n_hours*3600//30)):  # this works because the FMA dataset is all 30 second clips
    row = next(fma_dataset)
    name = row['audio']['path'].split('/')[-1].replace(".mp3", ".wav")
    scipy.io.wavfile.write(os.path.join(output_dir, name), 16000, (row['audio']['array']*32767).astype(np.int16))
    i += 1
    if i == n_hours*3600//30:
        break


# In[ ]:


print("Download pre-computed openWakeWord features for training and validation")

# training set (~2,000 hours from the ACAV100M Dataset)
# See https://huggingface.co/datasets/davidscripka/openwakeword_features for more information
os.system('wget https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/openwakeword_features_ACAV100M_2000_hrs_16bit.npy')

# validation set for false positive rate estimation (~11 hours)
os.system('wget https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/validation_set_features.npy')


# # Define Training Configuration

# For automated model training openWakeWord uses a specially designed training script and a [YAML](https://yaml.org/) configuration file that defines all of the information required for training a new wake word/phrase detection model.
# 
# It is strongly recommended that you review [the example config file](../examples/custom_model.yml), as each value is fully documented there. For the purposes of this notebook, we'll read in the YAML file to modify certain configuration parameters before saving a new YAML file for training our example model. Specifically:
# 
# - We'll train a detection model for the phrase "hey sebastian"
# - We'll only generate 5,000 positive and negative examples (to save on time for this example)
# - We'll only generate 1,000 validation positive and negative examples for early stopping (again to save time)
# - The model will only be trained for 10,000 steps (larger datasets will benefit from longer training)
# - We'll reduce the target metrics to account for the small dataset size and limited training.
# 
# On the topic of target metrics, there are *not* specific guidelines about what these metrics should be in practice, and you will need to conduct testing in your target deployment environment to establish good thresholds. However, from very limited testing the default values in the config file (accuracy >= 0.7, recall >= 0.5, false-positive rate <= 0.2 per hour) seem to produce models with reasonable performance.
# 

# In[ ]:


# Load default YAML config file for training
config = yaml.load(open("openwakeword/examples/custom_model.yml", 'r').read(), yaml.Loader)
config


# In[ ]:


print("Modify values in the config and save a new version")

config["target_phrase"] = ["simbad"]
config["model_name"] = config["target_phrase"][0].replace(" ", "_")
config["n_samples"] = 1000
config["n_samples_val"] = 1000
config["steps"] = 10000
config["target_accuracy"] = 0.6
config["target_recall"] = 0.25

config["background_paths"] = ['./audioset_16k', './fma']  # multiple background datasets are supported
config["false_positive_validation_data_path"] = "validation_set_features.npy"
config["feature_data_files"] = {"ACAV100M_sample": "openwakeword_features_ACAV100M_2000_hrs_16bit.npy"}

with open('my_model.yaml', 'w') as file:
    documents = yaml.dump(config, file)


# # Train the Model

# With the data downloaded and training configuration set, we can now start training the model. We'll do this in parts to better illustrate the sequence, but you can also execute every step at once for a fully automated process.

# In[ ]:


print("Step 1: Generate synthetic clips")
# For the number of clips we are using, this should take ~10 minutes on a free Google Colab instance with a T4 GPU
# If generation fails, you can simply run this command again as it will continue generating until the
# number of files meets the targets specified in the config file

os.system(f'{sys.executable} openwakeword/openwakeword/train.py --training_config my_model.yaml --generate_clips')


# In[ ]:


print("Step 2: Augment the generated clips")

os.system(f'{sys.executable} openwakeword/openwakeword/train.py --training_config my_model.yaml --augment_clips')


# In[ ]:


print("Step 3: Train model")

os.system(f'{sys.executable} openwakeword/openwakeword/train.py --training_config my_model.yaml --train_model')


# In[ ]:


# # Step 4 (Optional): On Google Colab, sometimes the .tflite model isn't saved correctly
# # If so, run this cell to retry

# # Manually save to tflite as this doesn't work right in colab
# def convert_onnx_to_tflite(onnx_model_path, output_path):
#     """Converts an ONNX version of an openwakeword model to the Tensorflow tflite format."""
#     # imports
#     import onnx
#     import logging
#     import tempfile
#     from onnx_tf.backend import prepare
#     import tensorflow as tf

#     # Convert to tflite from onnx model
#     onnx_model = onnx.load(onnx_model_path)
#     tf_rep = prepare(onnx_model, device="CPU")
#     with tempfile.TemporaryDirectory() as tmp_dir:
#         tf_rep.export_graph(os.path.join(tmp_dir, "tf_model"))
#         converter = tf.lite.TFLiteConverter.from_saved_model(os.path.join(tmp_dir, "tf_model"))
#         tflite_model = converter.convert()

#         logging.info(f"####\nSaving tflite mode to '{output_path}'")
#         with open(output_path, 'wb') as f:
#             f.write(tflite_model)

#     return None

# convert_onnx_to_tflite(f"my_custom_model/{config['model_name']}.onnx", f"my_custom_model/{config['model_name']}.tflite")


# After the model finishes training, the auto training script will automatically convert it to ONNX and tflite versions, saving them as `my_custom_model/<model_name>.onnx/tflite` in the present working directory, where `<model_name>` is defined in the YAML training config file. Either version can be used as normal with `openwakeword`. I recommend testing them with the [`detect_from_microphone.py`](https://github.com/dscripka/openWakeWord/blob/main/examples/detect_from_microphone.py) example script to see how the model performs!
