# https://github.com/tensorflow/tensorflow/issues/61886

x1 = tf.constant([[1., 2.], [3., 4.], [5., 6.]], shape=[3, 2])

class Model(tf.keras.Model):

  def __init__(self):
    super(Model, self).__init__()

  @tf.function(input_signature=[tf.TensorSpec(x1.shape, x1.dtype)])
  def call(self, x):
    a = tf.reshape(x, [3, 2, 1])
    b = tf.unstack(a, axis=1)
    c = tf.concat(b, 0)
    d = tf.reshape(c, [3, 2])
    return d

m = Model()
expected_value = m(x1)
print('keras model output:')
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
print('tflite model output:')
print(actual_value[0])
tf.lite.experimental.Analyzer.analyze(model_content=tflite_model)