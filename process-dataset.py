
import cv2 as cv
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
import tensorflow as tf
import numpy as np # linear algebra
import pandas as pd
import os
from frame_creator import Frame_creator

class pre_process(object):
    def __init__(self,file,video_name):
        self.file=file
        
        self.X=[]
        
        frames=Frame_creator(video_name)
        frames.extract()
        
    
    def read_file(self):
        
        self.data=pd.read_csv(self.file)
        list_img=self.data.Image_ID
        return list_img
    
    def create_dataset(self):
        
        lst=self.read_file()

        for name in lst:
            #print(name)
            img=cv.imread(os.path.join("..","working",name))
            if img is not None:
                img=cv.resize(img,(224,224),interpolation=cv.INTER_AREA)
                img=img/255.
                
                self.X.append(img)
            else:
                print("check",img)
        
        self.X=np.array(self.X)
        
        self.y=self.data.Class
        self.y=np.array(tf.one_hot(self.y,depth=3))
    
    def dataset_split(self):
        
        self.create_dataset()
        X_train, X_valid, y_train, y_valid = train_test_split(self.X, self.y, test_size=0.3, random_state=42)
        return X_train, X_valid, y_train, y_valid
    
    def class_weights(self):
        
        self.read_file()
        class_weights = compute_class_weight('balanced',np.unique(self.data.Class), self.data.Class)  # computing weights of different classes   

        return class_weights


