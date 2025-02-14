# https://github.com/tensorflow/tensorflow/issues/61150

import tensorflow as tf
import numpy as np
input_shape = [2, 2]
x1 = tf.constant([[0., 0.], [0., 0.]], shape=input_shape)

class Model(tf.keras.Model):

  def __init__(self):
    super(Model, self).__init__()
    self.w1 = tf.Variable([[0.], [1.]])
    self.m1 = tf.Variable([[1.], [1.]])
  @tf.function(input_signature=[tf.TensorSpec(x1.shape, x1.dtype)])
  def call(self, x1):
    x2 = tf.constant([1.], shape=[1])
    x3 = x1 + x2 #broadcast
    return tf.matmul(x3, self.w1) + self.m1

m = Model()
expected_value = m(x1)
print(expected_value.numpy())

converter = tf.lite.TFLiteConverter.from_keras_model(m)
tflite_model = converter.convert()

def _evaluateTFLiteModel(tflite_model, input_data):
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    for i in range(len(input_data)):
        interpreter.set_tensor(input_details[i]['index'], input_data[i])
    interpreter.invoke()
    output_data = [interpreter.get_tensor(output_details[i]['index'])
                   for i in range(len(output_details))]
    return output_data


actual_value = _evaluateTFLiteModel(tflite_model,[x1])
print(actual_value[0])
tf.lite.experimental.Analyzer.analyze(model_content=tflite_model)