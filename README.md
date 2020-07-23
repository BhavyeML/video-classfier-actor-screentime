# video-classfication Model to Calculate Actor-Screen time
  
  This repository contains a video classification model for prediction of screen time of actors in a video.

# Dataset
  
    I have token videos from popular TV cartoon series TOM and JERRY, and  I am calculating the screen time of Tom and Jerry from a ~5 min long
    video. I have extracted frames out of video at the rate 1 fps and thus in total we have 298 frames. These steps have been performed in frame_creator.py

    This script outputs  all the frame images using OpenCV and a csv file where each row contains name of the frame. We then have to manually label all the images.
    For the task of annotation we can eiter use online tools or we can manually label ~60 of the images and then for the rest of the images we can run a classifier
    to annotate them.
  
 # WorkFlow
  
    This repository contains three parts:
    1. Frame extraction and output csv file for manually labeling the frames
    2. Model Training on the extracted frames
    3. Using trained model to analysis screen-time in nnew Tom and Jerry Videos.


    
 # Data Pre-processing
  
     Once I completed the step 1. of frame extraction and annotation, we then use our extracted image to create the model. We first pass our data throug
     data_process.py to pre- pocess our dataset of images by reading them using OpenCV, normalising the pixel values b/w 0 and 1 and converting them into
     numpy array. I have also on-hot encoded the oup=tput labels with idx = O as 'Neither Tome nor Jerry', idx=1 as "jerry" idx=2 as "Tom".
     
 # Callback
   
    I have created a callback class present in video_callback.py that checks for a desired validation accuracy at each epoch's end. 
    This helps to stop training as soon as our model accuracy has been achieved. I have implmeneted it using tf.keras.callback class
    and we have kept desired validation accuracylow~80%, this is done beacuse we have very limited sample for training.
     
 # Model Training
     
     The training has been performed in video_train.py. I have used a pre-trained Mobile_net_v2 model from tensorflow hub for the purpose of feature extraction
     layer. I am fine tuning this layer with my dataset and on top of tis layer I have further used 3 dense layers to make prediction with 'relu' and 'softmax' 
     as activation uinit. I have also used dropout of value = 0.5. For compliation I have used Stochastic gradient descent optimizer  with lr =0.001 and my 
     loss function is categorical cross entropy. With callback passed in model training, we usually achieve our accuracy before 20 epochs. This misbalanced claases 
     values have also been taken into account while training There are other functions present in the class to visualize the model accuracy per epoch and to save the model.
     
 # Trained model for screen-time analysis
 
    To predict screen time of our desired actors(tom and jeery), I have created a predict_script.py which takes in an input video directory and and also the 
    trained model directory to make predictions on the new video. The results have been promising, i remember in one of the video I searched, it gave 90 secs 
    for Jerry while 159 secs for Tom.
     
     
    The predcitions can further be improved by adding more and more annotated data

  
  
    
   
   

  
