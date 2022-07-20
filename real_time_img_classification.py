import tensorflow as tf
import os
import numpy as np
import cv2

# model = tf.keras.models.load_model('tf_pose_classifier_v3.h5')

# img_path = '/home/nums/Pictures/Webcam/neg2.jpg'
# print(img_path)
# print('\n')
# img_height = 256
# img_width = 256
# img = tf.keras.utils.load_img(
#     img_path, target_size=(img_height, img_width)
# )
# img_array = tf.keras.utils.img_to_array(img)
# img_array = tf.expand_dims(img_array, 0) # Create a batch

# predictions = model.predict(img_array)
# print("predictions")
# print(predictions)
# score = tf.nn.softmax(predictions[0])
# print("score", score)

# class_names = ['neg','pos']
# print(
#     "This image most likely belongs to {} with a {:.2f} percent confidence."
#     .format(class_names[np.argmax(score)], 100 * np.max(score))
# )

# I think the model just learned to always predict negative. Need to trainf for higher accuracy. (83% is 5/6)
# Fixed by just training longer now add real-time component

def predictImageClass(img):
    model = tf.keras.models.load_model('tf_pose_classifier_v5.h5')

    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    print("predictions")
    print(predictions)
    score = tf.nn.softmax(predictions[0])
    print("score", score)

    class_names = ['zone 1','zone 2', 'zone 3', 'zone 4']
    print(
        "{} {:.2f} % confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )

# img = cv2.imread("./saved_image_1.jpg")
# resized = cv2.resize(img, (256, 256))
# predictImageClass(resized)


# define a video capture object
vid = cv2.VideoCapture(2)
  
# while(True):
for i in range(20):
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    # Display the resulting frame
    cv2.imshow('frame', frame)
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    resized = cv2.resize(frame, (256, 256))
    predictImageClass(resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        # resized = cv2.resize(frame, (256, 256))
        # cv2.imwrite('testimage.jpg', frame)
        # predictImageClass(resized)
        break
    if i == 15:
        cv2.imwrite('testimage.jpg', resized)
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()