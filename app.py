import cv2
import numpy as np
from keras.models import load_model
import os
import time
import re
import sys
import pyttsx3

pathname = os.path.dirname(sys.argv[0])

print('sys.argv[0] =', sys.argv[0], "pathname", pathname)    

# Load the model
model = load_model(f'{pathname}/model.savedmodel')

# CAMERA can be 0 or 1 based on default camera of your computer.


# Grab the labels from the labels.txt file. This will be used later.
labels = open(f'{pathname}/labels.txt', 'r').readlines()

engine = pyttsx3.init()

while True:
    time.sleep(5)

    camera = cv2.VideoCapture(0)
    # Grab the webcameras image.
    ret, image = camera.read()
    camera.release()
    #print (image.shape)
    # Resize the raw image into (224-height,224-width) pixels.
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    # Show the image in a window
    #cv2.imshow('Webcam Image', image)
    # Make the image a numpy array and reshape it to the models input shape.
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    # Normalize the image array
    image = (image / 127.5) - 1
    # Have the model predict what the current image is. Model.predict
    # returns an array of percentages. Example:[0.2,0.8] meaning its 20% sure
    # it is the first label and 80% sure its the second label.
    probabilities = model.predict(image)
    # Print what the highest value probabilitie label
    max_prob = np.max(probabilities)
    label = re.sub(r'\d+\s+', '', labels[np.argmax(probabilities)]).strip()

    print (label, f"proba: {max_prob}")
    if label != "empty" and max_prob > 0.9:
        
        #os.system(f"cd voice && {label}.mp3")
        
        os.system(f"festival --tts {pathname}/voice/{label}.txt")
        #engine.say(f"{label}")
        #engine.runAndWait()

    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)
    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break
    
    
    

#cv2.destroyAllWindows()
