# ReconocimientoDigitos

https://colab.research.google.com/

paso 1:

try:

  %tensorflow_version 2.x

except Exception:

  pass

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow import keras

import numpy as np

import matplotlib.pyplot as plt

import random

print(tf.__version__)

paso 2:

mnist = keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

paso 3:

train_images = train_images / 255.0

test_images = test_images / 255.0

print('Pixels are normalized')

paso 4:

plt.figure(figsize=(10,10))

for i in range(25):

  plt.subplot(5,5,i+1)
  
  plt.xticks([])
  
  plt.yticks([])
  
  plt.grid(False)
  
  plt.imshow(train_images[i], cmap=plt.cm.gray)
  
  plt.xlabel(train_labels[i])

plt.show()

paso 5:

model = keras.Sequential([

  keras.layers.InputLayer(input_shape=(28, 28)),
  
  keras.layers.Reshape(target_shape=(28, 28, 1)),
  
  keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation=tf.nn.relu),
  
  keras.layers.MaxPooling2D(pool_size=(2, 2)),
  
  keras.layers.Flatten(),
  
  keras.layers.Dense(10, activation=tf.nn.softmax)
  
])

model.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',
              
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)

paso 6:

model.summary()

paso 7:

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)

paso 8:

def get_label_color(val1, val2):

  if val1 == val2:
  
    return 'black'
    
  else:
  
    return 'red'

predictions = model.predict(test_images)

prediction_digits = np.argmax(predictions, axis=1)

plt.figure(figsize=(18, 18))

for i in range(100):

  ax = plt.subplot(10, 10, i+1)
  
  plt.xticks([])
  
  plt.yticks([])
  
  plt.grid(False)
  
  image_index = random.randint(0, len(prediction_digits))
  
  plt.imshow(test_images[image_index], cmap=plt.cm.gray)
  
  ax.xaxis.label.set_color(get_label_color(prediction_digits[image_index],\
  
                                           test_labels[image_index]))
                                           
  plt.xlabel('Predicted: %d' % prediction_digits[image_index])
  
plt.show()

paso 9:

converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_float_model = converter.convert()

float_model_size = len(tflite_float_model) / 1024

print('Float model size = %dKBs.' % float_model_size)

paso 10:

converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_quantized_model = converter.convert()

quantized_model_size = len(tflite_quantized_model) / 1024

print('Quantized model size = %dKBs,' % quantized_model_size)

print('which is about %d%% of the float model size.'\

      % (quantized_model_size * 100 / float_model_size))

paso 11:

def evaluate_tflite_model(tflite_model):

  interpreter = tf.lite.Interpreter(model_content=tflite_model)
  
  interpreter.allocate_tensors()
  
  input_tensor_index = interpreter.get_input_details()[0]["index"]
  
  output = interpreter.tensor(interpreter.get_output_details()[0]["index"])
  
  prediction_digits = []
  
  for test_image in test_images:
    
    test_image = np.expand_dims(test_image, axis=0).astype(np.float32)
    
    interpreter.set_tensor(input_tensor_index, test_image)

    interpreter.invoke()

    digit = np.argmax(output()[0])
    
    prediction_digits.append(digit)

  accurate_count = 0
  
  for index in range(len(prediction_digits)):
  
    if prediction_digits[index] == test_labels[index]:
    
      accurate_count += 1
      
  accuracy = accurate_count * 1.0 / len(prediction_digits)

  return accuracy
  
float_accuracy = evaluate_tflite_model(tflite_float_model)

print('Float model accuracy = %.4f' % float_accuracy)

quantized_accuracy = evaluate_tflite_model(tflite_quantized_model)

print('Quantized model accuracy = %.4f' % quantized_accuracy)

print('Accuracy drop = %.4f' % (float_accuracy - quantized_accuracy))

paso 12:

f = open('mnist.tflite', "wb")

f.write(tflite_quantized_model)

f.close()

from google.colab import files

files.download('mnist.tflite')

print('`mnist.tflite` has been downloaded')
