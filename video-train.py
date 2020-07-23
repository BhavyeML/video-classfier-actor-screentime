# %% [code]
import cv2 as cv
import math
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split
import matplotlib.pylab as plt
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
import numpy as np # linear algebra
import pandas as pd
import os
import csv
import process_dataset
from process_dataset import pre_process
import video_callback
from video_callback import mycallback

# %% [code]
class model_net(object):
    def __init__(self,file,video_name):
        
        self.feature_extractor_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2"
        self.pre_model=pre_process(file,video_name)
        
        
        self.IMG_SIZE = 224
        #self.BATCH_SIZE = 32
  
    def dataset(self):
        
        self.x,self.x_val,self.y,self.y_val= self.pre_model.dataset_split()
        
    
    def base_arch(self):
        
        self.feature_extractor_layer = hub.KerasLayer(self.feature_extractor_url,
                                         input_shape=(224,224,3))
        self.feature_extractor_layer.trainable = True
        #print("Number of layers in the base model: ", len(self.feature_extractor_layer))


    
    def model_arch(self):
        
        self.base_arch()
        
        self.model = tf.keras.Sequential([
            self.feature_extractor_layer,
            tf.keras.layers.Dense(1024,activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(512,activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(256,activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(3,activation='softmax')
        ])
        
        self.model.summary()
        
    
    def train(self):
        
        self.dataset()
        self.model_arch()
        class_weights=self.pre_model.class_weights()
        callbacks=mycallback()
        from tensorflow.keras.optimizers import SGD
        sgd = SGD(lr=0.001, momentum=0.9, nesterov=True)
        
        self.model.compile(
                     optimizer=sgd,
                     loss='categorical_crossentropy',
                     metrics=['accuracy']
                    )
    
        device_name=tf.test.gpu_device_name()
        
        if "GPU" not in device_name:
            print("Training on Cpu")
            config=tf.compat.v1.ConfigProto()
            config.gpu_options.allow_growth=True
            history = self.model.fit(x=self.x,y=self.y,epochs=50,batch_size=32,shuffle=True,verbose=2,validation_data=(self.x_val,self.y_val),callbacks=[callbacks])

        else:
            
            print("Training on Gpu")
            with tf.device('/gpu:0'):
                        history = self.model.fit(x=self.x,y=self.y,epochs=50,batch_size=32,shuffle=True,verbose=2,validation_data=(self.x_val,self.y_val),callbacks=[callbacks])

        
        
        self.visualize_train(history)
        
        self.save_model()
        
    def visualize_train(self,history):
        
        acc=history.history['accuracy']
        val_acc=history.history['val_accuracy']
        epochs=range(len(acc))
        plt.plot(epochs,acc)
        plt.plot(epochs,val_acc)
        plt.show()
        
    def save_model(self):
        #self.model.save("final_model.h5")
        tf.keras.models.save_model(self.model,'model_train')

# %% [code]
if __name__=="__main__":
    model=model_net("../input/acttorscreen/mapping.csv","../input/acttorscreen/Tom and jerry.mp4")
    model.train()

# %% [code]
