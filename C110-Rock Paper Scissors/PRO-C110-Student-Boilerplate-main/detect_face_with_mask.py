# import the opencv library
import cv2
import numpy as np #used for numerical calculations in an array
import tensorflow as tf #used for classification of images ("Keras is part of tensorflow")
model = tf.keras.models.load_model("C:/Users/vrish_fl8o8kc/Downloads/C110-Rock Paper Scissors/PRO-C110-Student-Boilerplate-main/converted_keras/keras_model.h5")
# define a video capture object
vid = cv2.VideoCapture(0)
  
while(True):
      
    # Capture the video frame by frame
    ret, frame = vid.read() #ret is holding boolean val(true or false) & frame is storing fram
    img = cv2.resize(frame,(224,224))
    i1 = np.array(img,dtype=np.float32)#this is a numpy array which can be changed with maths
#line14-converts into numpy arrays and datatype is float32
    i2 = np.expand_dims(i1, axis=0)#axis(0)means x axis. axis(1) is y axis.
    n_img = i2/255.0
    prediction = model.predict(n_img) #predicting the decimal value of the colour
    predict_class = np.argmax(prediction,axis=1)#argmax returns the maximum value of the array
    print(predict_class)
    # Display the resulting frame
    cv2.imshow('frame', frame)
      
    # Quit window with spacebar
    key = cv2.waitKey(1)
    
    if key == 32:
        break
  
# After the loop release the cap object
vid.release()

# Destroy all the windows
cv2.destroyAllWindows()