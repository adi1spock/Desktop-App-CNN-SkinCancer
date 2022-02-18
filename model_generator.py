import cv2
from keras.preprocessing.image import ImageDataGenerator
image_gen = ImageDataGenerator(rotation_range=30, 
                               width_shift_range=0.1, 
                               height_shift_range=0.1, 
                               rescale=1/255, 
                               shear_range=0.2, 
                               zoom_range=0.2, 
                               horizontal_flip=True, 
                               fill_mode='nearest'
                              )
image_gen.flow_from_directory('dataset/train_set')
image_gen.flow_from_directory('dataset/test_set')
image_shape = (600,450,3)
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3,3),input_shape=(600,450,3), activation='relu',))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=64, kernel_size=(3,3),input_shape=(600,450,3), activation='relu',))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=64, kernel_size=(3,3),input_shape=(600,450,3), activation='relu',))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
train_image_gen = image_gen.flow_from_directory('dataset/train_set',
                                               target_size=(600,450),
                                               batch_size=16,
                                               class_mode='binary') 
test_image_gen = image_gen.flow_from_directory('dataset/test_set',
                                               target_size=(600,450),
                                               batch_size=16,
                                               class_mode='binary')                                               
results = model.fit_generator(train_image_gen,epochs=2,
                              steps_per_epoch=15,
                              validation_data=test_image_gen,
                             validation_steps=12)
model.save('heloo.h5')
model.load('heloo.h5')


import numpy as np
from keras.preprocessing import image


test_file = 'C:\\Users\\Aditya\\Downloads\\real\\dataset\\image'
test_img = image.load_img(test_file, target_size=(600, 450))


test_img = image.img_to_array(test_img)

test_img = np.expand_dims(test_img, axis=0)
test_img = test_img/255
prediction_prob = model.predict(test_img)
if prediction_prob>=0.9:
    print("you have very chance of having cancer")
else:
    print("you have very low chance of having cancer") 
                                               
            