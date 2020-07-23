# %% [code]
import cv2 as cv
import math
import tensorflow as tf
import numpy as np
import os

class predict_screentime(object):
    
    def __init__(self,link,model_file):
        self.link=link
        self.model = tf.keras.models.load_model(model_file)
        self.lst=[]
        self.X=[]
        
    def get_frames(self):
        
        cap = cv.VideoCapture(self.link)
        
        FrameRate=cap.get(5)
        idx=0
        
        
        while(cap.isOpened()):
            
            
            frameId=cap.get(1)
            
            ret,frame= cap.read()
            
            if(ret!= True):
                break
                
            if(frameId%math.floor(FrameRate)==0):
                filename="frame{}.jpg".format(idx)
                idx+=1
                self.lst.append(filename)
                cv.imwrite(filename,frame)
        
                
        cap.release()
        
    def pre_processing(self):
        self.get_frames()
        for name in self.lst:
            #print(name)
            img=cv.imread(os.path.join("..","working",name))
            if img is not None:
                img=cv.resize(img,(224,224),interpolation=cv.INTER_AREA)
                img=img/255.
                self.X.append(img)
            else:
                print("check",img)
        
        self.X=np.array(self.X)
        
    def predict(self):
        
        self.pre_processing()
        self.predictions=self.model.predict(self.X)
        self.prediction_idx=np.argmax(self.predictions,axis=1)
       
    def calculate_time(self):
        
        self.predict()
        print("The screen time of JERRY is", self.prediction_idx[self.prediction_idx==1].shape[0], "seconds")
        print("The screen time of TOM is", self.prediction_idx[self.prediction_idx==2].shape[0], "seconds")
        



if __name__=="__main__":
    actors=predict_screentime("../input/acttorscreen/Tom and Jerry 3.mp4","../input/model-weight/model_train")
    actors.calculate_time()