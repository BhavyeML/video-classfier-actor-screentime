
import numpy as np # linear algebra


class Frame_creator(object):
    
    def __init__(self, link):
        
        self.link =link
        self.lst=[]
    def extract(self):
        import cv2 as cv
        import math
        
        cap = cv.VideoCapture(self.link)
        
        FrameRate=cap.get(5)
        idx=0
        
        
        while(cap.isOpened()):
            
            
            frameId=cap.get(1)
            
            ret,frame= cap.read()
            
            if(ret!= True):
                break
                
            if(frameId%math.floor(FrameRate)==0):
                #print("in-loop")
                filename="frame{}.jpg".format(idx)
                idx+=1
                self.lst.append(filename)
                cv.imwrite(filename,frame)
        
                
        cap.release()
        
    def csv_logging(self):
        import csv
        self.extract()
        fieldname=['Image_ID']
        with open("label_file.csv",mode='w') as csv_file:
            csv_writer=csv.writer(csv_file,delimiter='\t')
            csv_writer.writerow(fieldname)
                 
            for img in self.lst:
                csv_writer.writerow(img)

                
if __name__=="__main__":
    frames=Frame_creator("../input/acttorscreen/Tom and jerry.mp4")
    frames.csv_logging()