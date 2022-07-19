import tensorflow as tf
import os
import numpy as np

model = tf.keras.models.load_model('tf_pose_classifier_v5.h5')

img_path = os.path.join(os.getcwd(), 'data/zone 4/105.jpg')
print(img_path)
print('\n')
img_height = 256
img_width = 256
img = tf.keras.utils.load_img(
    img_path, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
print(img_array.shape)
print(np.array(img_array).dtype)

# img_array = tf.expand_dims(img_array, 0) # Create a batch

# predictions = model.predict(img_array)
# print("predictions")
# print(predictions)
# score = tf.nn.softmax(predictions[0])
# print("score", score)

# class_names = ['zone 1','zone 2', 'zone 3', 'zone 4']
# print(
#     "This image most likely belongs to {} with a {:.2f} percent confidence."
#     .format(class_names[np.argmax(score)], 100 * np.max(score))
# )