# https://github.com/tensorflow/tensorflow/issues/60892
import tensorflow as tf
import numpy as np

x1 = tf.constant([100., 100.], shape=[1, 2])

class Model(tf.keras.Model):

  def __init__(self):
    super(Model, self).__init__()

  @tf.function(input_signature=[tf.TensorSpec(x1.shape, x1.dtype)])
  def call(self, x):
    return tf.math.softplus(x)

# Initializing the model
m = Model()

m(x1)
print('Keras mode output: ', m(x1))

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

print('Lite mode output: ', _evaluateTFLiteModel(tflite_model,[x1])[0])