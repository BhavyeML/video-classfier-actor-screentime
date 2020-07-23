import tensorflow as tf 

class mycallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epochs,logs={}):
        if(logs.get('val_accuracy')>0.78):
            print("Achieved desired accuracy")
            self.model.stop_training=True
            