# Example: PyTorch -> ONNX -> TensorFlow -> TF.js
import torch
import tensorflow as tf
import onnx
from onnx_tf.backend import prepare


onnx_model = onnx.load("models/best_model.onnx")
tf_rep = prepare(onnx_model)
tf_rep.export_graph("best_model_tf")