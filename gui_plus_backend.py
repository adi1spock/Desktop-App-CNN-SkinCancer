from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
import numpy as np
from keras.preprocessing import image
model=Sequential()
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
model.load_weights('heloo.h5')




#tkinter appp
import tkinter as tk
from PIL import ImageTk,Image
root =tk.Tk()
e=tk.Entry(root,width=200)
e.grid(row=1,column=0)
f=tk.Entry(root,width=200)
f.grid(row=7,column=0)
def myClick():
    img_path=e.get()
    print(img_path)
    myLabel=tk.Label(root,text="scan is underway")
    myLabel.grid(row=4,column=0)
    test_file = "C:\\Program Files (x86)\\new\\dataset\\image\\"+img_path
    my_img =ImageTk.PhotoImage(Image.open(test_file))
    my_label=tk.Label(image=my_img)
    my_label.grid(row=3,column=0)
    test_img = image.load_img(test_file, target_size=(600,450))
    test_img = image.img_to_array(test_img)
    test_img = np.expand_dims(test_img, axis=0)
    test_img = test_img/255
    prediction_prob = model.predict(test_img)
    if prediction_prob>=0.9:
        message="you have very high chance of having cancer"
        f.insert(0,message)
    else:
        message="***you have very low chance of having cancer******   :) :) don't worry"
        f.insert(0,message)
		                   
myButton=tk.Button(root,text="click to check for skin cancer",command=myClick,fg="yellow",bg="gray")
myButton.grid(row=2,column=0)
mylabel=tk.Label(root,text="enter the name of the image")
mylabel.grid(row=0,column=0)
button_quit=tk.Button(root,text="EXIT",command=root.quit)
button_quit.grid(row=8,column=0)
root.mainloop()



#


