
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow.compat.v1 as tf
import mss
import os
import time


def get_Note(image,sess):
    infrencing = sess.run(output_tensor, {'x:0': image})
    return infrencing
## Loading model
sess=tf.InteractiveSession()
frozen_graph="./frozen_models/simple_frozen_graph.pb"
with tf.gfile.GFile(frozen_graph, "rb") as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
sess.graph.as_default()
tf.import_graph_def(graph_def)
# Frozen model inputs: 
# [<tf.Tensor 'x:0' shape=(None, None, None, 3) dtype=float32>]
# Frozen model outputs: 
# [<tf.Tensor 'Identity:0' shape=(None, None, None, 8) dtype=float32>]  
input_tensor = sess.graph.get_tensor_by_name("x:0") 
output_tensor = sess.graph.get_tensor_by_name("Identity:0")        



def GetScreenShot(monitor = 1):
    with mss.mss() as mss_instance:  # Create a new mss.mss instance
        #monitor_1 = mss_instance.monitors[1]  # Identify the display to capture
        monitor_1 = {"top": 1000, "left": 750, "width": 150, "height": 300} # Identify the display to capture
        screenshot = mss_instance.grab(monitor_1)  # Take the screenshot
    open_cv_image = np.array(screenshot) 
    grayscaleImage = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    ScreenShot  = cv2.cvtColor(grayscaleImage, cv2.COLOR_GRAY2RGB)    
    #cv2.imshow("Image", ScreenShot)
    #cv2.waitKey(0)
    testimage=cv2.resize(ScreenShot,(224,224,))
    testimage = np.expand_dims(testimage,axis = 0)
    return testimage
    
    


    
    


for i in range(5):
    
    start = time.clock()
    img = GetScreenShot()
    print(get_Note(img,sess))
    print("Time: ", start - time.clock())
    
    
