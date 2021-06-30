import tensorflow as tf 
from tensorflow.keras.preprocessing import image
import numpy as np 
import matplotlib.pyplot as plt
import sys


# this function takes an image and then runs the predict method on the image and plot the results 
def infer(fname):
	input_shape=(256,256,3) #performing basic preprocessing of image data before prediction
	img=image.load_img(fname,target_size=(256,256))
	x=image.img_to_array(img)
	x=x/255
	x=np.expand_dims(x,axis=0)
	model=tf.keras.models.load_model("give full path to brain_MRI.h5")
	pred=model.predict(x)
	plt.figure(figsize=(15,15))
	plt.subplot(1,2,1)
	plt.imshow(np.squeeze(x))
	plt.title("Image")
	plt.subplot(1,2,2)
	plt.imshow(np.squeeze(pred))
	plt.title("Predicted Mask")
	plt.show()

if __name__=='__main__':

	file_name=sys.argv[1] #getting the filename from terminal 
	infer(file_name)




