import os
import random
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use("ggplot")
%matplotlib inline
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator




path = "/kaggle/input/lgg-mri-segmentation/kaggle_3m/"  #the path to the dataset



def get_data(path): # this function takes the path and then convert the directories into images and masks
    dirs = [] # var's to store the directories without full path,images and masks 
    images = []
    masks = []
    for dirname, _, filenames in os.walk(path): 
        for filename in filenames:
            if 'mask' in filename:
                dirs.append(dirname.replace(path,'')) #appending the directories corresponding to each 
                masks.append(filename)
                images.append(filename.replace('_mask',''))
    return dirs,images,masks

dirs,images,masks=get_data(path)
df = pd.DataFrame({'directory':dirs, 'images': images, 'masks': masks}) # creating a dataframe with columns directory,images,masks and they will contain the directory names for each
df.head()

def plot_images(path,df,k):
    imagePath = os.path.join(path, df['directory'].iloc[idx], df['images'].iloc[idx])
    maskPath = os.path.join(path, df['directory'].iloc[idx], df['masks'].iloc[idx]) 
    image = cv2.imread(imagePath)
    mask = cv2.imread(maskPath)
    plt.figure(figsize=(15,15))
    plt.subplot(1,4,1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.subplot(1,4,2)
    plt.imshow(mask)
    plt.title('Original Mask')
    plt.show()

for i in range(4):
    k = np.random.randint(0, len(df))
    plot_images(path,df,k)



df['image_path'] = path + df['directory'] + '/' + df['images'] #creating a new column in dataframe that will have the full path to images and masks
df['mask_path'] = path + df['directory'] + '/' + df['masks'] 
train , test = train_test_split(df, test_size=0.20, random_state=16) # creating a train and test set

EPOCHS = 10    # running the model for just 10  epoch as the accuracy in the first epoch for both train and validation set is greater than 97% and each epoch takes around 17 mins 
#setting the batch size ,image height and width
BATCH_SIZE = 32
IMAGE_HEIGHT=256
IMAGE_WIDTH=256
Channels=3


def augment(df,batch_size,IMAGE_HEIGHT,IMAGE_WIDTH,image_gen,mask_gen):   #creating a function to perform data augmentation , we are passing the dataframe ,the generators ,the new image height and width
    aug_images=image_gen.flow_from_dataframe(dataframe=df,x_col='image_path',
                                              batch_size=batch_size,class_mode=None,
                                               seed=42,
                                              target_size=(IMAGE_HEIGHT,IMAGE_WIDTH),
                                              color_mode='rgb')
    aug_masks=mask_gen.flow_from_dataframe(dataframe=df,x_col='mask_path',
                                              batch_size=batch_size,class_mode=None,
                                             seed=42,
                                              target_size=(IMAGE_HEIGHT,IMAGE_WIDTH),
                                              color_mode='grayscale')
    return aug_images,aug_masks

#creating image data generator for augmenting the data
image_gen = ImageDataGenerator(rescale=1./255., rotation_range=0.2,
                        width_shift_range=0.01,
                        height_shift_range=0.01,
                        shear_range=0.01,
                        zoom_range=0.01,
                        horizontal_flip=True,
                        fill_mode='nearest')
mask_gen = ImageDataGenerator(rescale=1./255.,rotation_range=0.2,
                        width_shift_range=0.01,
                        height_shift_range=0.01,
                        shear_range=0.01,
                        zoom_range=0.01,
                        horizontal_flip=True,
                        fill_mode='nearest')



#passing the train set to the augment function
train_images,train_masks=augment(train,BATCH_SIZE,IMAGE_HEIGHT,IMAGE_WIDTH,image_gen,mask_gen)
#creating image data generators for validation data , we are just performing rescaling and not anything else 
val_image_gen = ImageDataGenerator(rescale=1./255.)
val_mask_gen = ImageDataGenerator(rescale=1./255.)
#passing the validation set to the augment function
val_images,val_masks=augment(test,BATCH_SIZE,IMAGE_HEIGHT,IMAGE_WIDTH,val_image_gen,val_mask_gen)


def data_generator(image_gen, mask_gen): #creating a generator which gives image and mask
    for img, mask in zip(image_gen, mask_gen):
        yield img, mask


train_gen = data_generator(train_images, train_masks)
valid_gen = data_generator(val_images, val_masks)



#the below code is the architecture for UNET

def conv2d_block(input_tensor,num_filters,kernel_size=(3,3)):
    x=input_tensor
    for i in range(2):
        x=tf.keras.layers.Conv2D(filters=num_filters,kernel_size=kernel_size,padding='same')(x)
        x=tf.keras.layers.Activation('relu')(x)
    return x
def encoder_block(input_tensor,num_filters=64,pool_size=(2,2),dropout=0.2):
  # adds two conv bolck and downsamples the output
    f=conv2d_block(input_tensor,num_filters=num_filters)
    p=tf.keras.layers.MaxPooling2D(pool_size=pool_size)(f)
    p=tf.keras.layers.Dropout(dropout)(p)
    return f,p


def encoder(input_tensor): #Sequence of convolution combined with max pooling results in down sampling. In down sampling Size of image is reduced which means we can observe larger portion of image in a single convolution operation. Down sampling is a good approach for identifying what is present in the image. but for identifying where the object is we need to use upsampling. 
    f1,p1=encoder_block(input_tensor,num_filters=64,pool_size=(2,2))
    f2,p2=encoder_block(p1,num_filters=128,pool_size=(2,2))
    f3,p3=encoder_block(p2,num_filters=256,pool_size=(2,2))
    f4,p4=encoder_block(p3,num_filters=512,pool_size=(2,2))
    return p4,(f1,f2,f3,f4)


def bottleneck(input_tensor):
    bottle_neck=conv2d_block(input_tensor,num_filters=1024)
    return bottle_neck
def decoder_block(inputs,conv_output,num_filters=64,kernel_size=(3,3),strides=3,dropout=0.3):
    u=tf.keras.layers.Conv2DTranspose(num_filters,kernel_size=kernel_size,strides=strides,padding='same')(inputs)
    c=tf.keras.layers.concatenate([u,conv_output])
    c=tf.keras.layers.Dropout(dropout)(c)
    c=conv2d_block(c,num_filters,kernel_size=kernel_size)
    return c
def decoder(inputs, convs, output_channels): #Up Sampling: It is just opposite of down sampling. We go from low resolution to high resolution. For up sampling UNet uses transposed covolution which is achieved by taking transpose of filter kernels and reversing the process of convolution.
    f1, f2, f3, f4 = convs
    c6 = decoder_block(inputs, f4, num_filters=512, kernel_size=(3,3), strides=(2,2), dropout=0.3)
    c7 = decoder_block(c6, f3, num_filters=256, kernel_size=(3,3), strides=(2,2), dropout=0.3)
    c8 = decoder_block(c7, f2, num_filters=128, kernel_size=(3,3), strides=(2,2), dropout=0.3)
    c9 = decoder_block(c8, f1, num_filters=64, kernel_size=(3,3), strides=(2,2), dropout=0.3)
    outputs = tf.keras.layers.Conv2D(output_channels, (1, 1), activation='sigmoid')(c9)
    return outputs

OUTPUT_CHANNELS = 1

def unet(): #defining the unet 
    inputs = tf.keras.layers.Input(shape=(256,256,3,)) #setting the input shape
    encoder_output, convs = encoder(inputs)
    bottle_neck = bottleneck(encoder_output)
    outputs = decoder(bottle_neck, convs, output_channels=OUTPUT_CHANNELS)
    model = tf.keras.Model(inputs=inputs, outputs=outputs) #creating the model
    return model

model = unet()
model.summary()
model.compile(optimizer='adam', loss="binary_crossentropy", metrics=["accuracy"]) # the loss is binary as we have to check if flair is  present or not

# setting the call backs and train_step and valid_step
callbacks = [
    EarlyStopping(patience=10, verbose=1),   
]


train_step = train_images.n/BATCH_SIZE #to get the step size we are dividing the total number wih the batch size
val_step = val_images.n/BATCH_SIZE


results = model.fit(train_gen,
                    steps_per_epoch=train_step,
                    batch_size=BATCH_SIZE,
                    epochs=10,
                    callbacks=callbacks,
                    validation_data=val_step,
                   validation_steps=STEP_SIZE_VALID)

model.save("brain_MRI.h5") # saving the model 

model=tf.keras.models.load_model("brain_MRI.h5") #loading the model to test

eval_results = model.evaluate(valid_gen, steps=STEP_SIZE_VALID, verbose=1) #testing the model 


###
def test_model(idx,path,test,model): #function to plot the predictions from test set 
    imagePath = os.path.join(path, test['directory'].iloc[idx], test['images'].iloc[idx])
    maskPath = os.path.join(path, test['directory'].iloc[idx], test['masks'].iloc[idx])
    
    image = cv2.imread(imagePath)
    mask = cv2.imread(maskPath)
    
    img = cv2.resize(image ,(256,256))
    img = img / 255
    img = img[np.newaxis, :, :, :]
    pred=model.predict(img)

    plt.figure(figsize=(12,12))
    plt.subplot(1,4,1)
    plt.imshow(np.squeeze(img))
    plt.title('Original Image')
    plt.subplot(1,4,2)
    plt.imshow(mask)
    plt.title('Original Mask')
    plt.subplot(1,4,3)
    plt.imshow(np.squeeze(pred))
    plt.title('Prediction')
    
   
    plt.show()
 


 for i in range(4):
    idx = np.random.randint(0, len(test))
    test_model(idx,path,test,model)


