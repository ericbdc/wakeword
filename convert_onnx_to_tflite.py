#!/usr/bin/env python

import os
import onnx
import logging
import tempfile
from onnx_tf.backend import prepare
import tensorflow as tf
import argparse


# Parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_folder",
    help="The name of the folder of a specific model to load",
    type=str,
    default="simbad",
    required=False,
)
parser.add_argument(
    "--model_name",
    help="The name of a specific model to load",
    type=str,
    default="",
    required=False,
)

args = parser.parse_args()


def convert_onnx_to_tflite(onnx_model_path, output_path):
    """Converts an ONNX version of model to the Tensorflow tflite format."""

    onnx_model = onnx.load(onnx_model_path)
    tf_rep = prepare(onnx_model, device="CPU")

    with tempfile.TemporaryDirectory() as tmp_dir:
        path_to_graph = os.path.join(tmp_dir, "tf_forzen_graph")
        tf_rep.export_graph(path_to_graph)

        converter = tf.lite.TFLiteConverter.from_saved_model(path_to_graph)
        tflite_model = converter.convert()

        logging.info(f"####\nSaving tflite mode to '{output_path}'")
        with open(output_path, "wb") as f:
            f.write(tflite_model)


if __name__ == "__main__":
    convert_onnx_to_tflite(
        f"{args.model_folder}/{args.model_name}.onnx",
        f"{args.model_folder}/{args.model_name}.tflite",
    )
