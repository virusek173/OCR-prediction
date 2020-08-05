from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
from keras.models import load_model
from utils import save_image

import keras
import cv2
import tensorflow as tf 

pretict_model_path = './final_model.h5'
first_char_code = 33

def load_image(image):
  treshold_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 20)
  image = (255-treshold_image)
  border_size = 8
  image = cv2.copyMakeBorder(image, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=0)
  image = cv2.resize(image, (28, 28)) 
  save_image(0, image, 'out_test')
  image = image.reshape(1, 28, 28, 1)
  image = image.astype('float32')
  image = image / 255.0 

  return image
 
def predict_image(model, image):
  image = load_image(image)
  digit = model.predict_classes(image)

  return digit[0]


model = tf.keras.models.load_model(pretict_model_path)
filename = 'cropped_images/50/1_20200516_160141447_50_203.jpg'
image = cv2.imread(filename, 0)
predicted_digit = predict_image(model, image)
predicted_digit += first_char_code
predicted_digit = chr(predicted_digit)
print('predicted_digit: {}'.format(predicted_digit))