# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 13:22:57 2020

@author: Marvin
"""

import tensorflow as tf
import pathlib
# import pandas as pd
import IPython.display as display
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator


AUTOTUNE = tf.data.experimental.AUTOTUNE

data_dir = "./images"
data_dir = pathlib.Path(data_dir)

train_images = list(data_dir.glob("train*.jpg"))
train_image_count = len(list(data_dir.glob('train*.jpg')))
test_image_count = len(list(data_dir.glob("test*jpg")))

train_dir = "train.csv"
test_dir = "test.csv"

#test_labels = pd.read_csv(test_dir)

train_labels = pd.read_csv(train_dir)
CLASS_NAMES = np.array(train_labels.columns)[1:]
print(CLASS_NAMES)
def load_data(data_dir):   
    train_target = train_labels[["healthy", "multiple_diseases", "rust", "scab"]][:1457]
    train_data = train_labels["image_id"][:1457]
    train_data = train_data.transform(lambda x: str(data_dir/x)+".jpg")
    
    val_target = train_labels[["healthy", "multiple_diseases", "rust", "scab"]][1457:]
    val_data = train_labels["image_id"][1457:]
    val_data = val_data.transform(lambda x: str(data_dir/x)+".jpg")
    print(train_data.shape, val_data.shape)
    train_list_ds = tf.data.Dataset.from_tensor_slices((train_data.values, train_target.values))
    val_list_ds = tf.data.Dataset.from_tensor_slices((val_data.values, val_target.values))
    
    return train_list_ds, val_list_ds

def show_data(list_ds):
    for feat, targ in list_ds.take(3):
        print ('Features: {}, Target: {}'.format(feat, targ))

def show_img(train_images):
    for image_path in train_images[:3]:
        img = Image.open(str(image_path))
        img.thumbnail((256, 256))
        display.display(img)
        
def decode_img(img):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_jpeg(img, channels=3)
  # Use `convert_image_dtype` to convert to floats in the [0,1] range.
  img = tf.image.convert_image_dtype(img, tf.float32)
  # resize the image to the desired size.
  return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])

def process_path(file_path, label):
    #print(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label

image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
BATCH_SIZE = 64
IMG_HEIGHT = 224
IMG_WIDTH = 224
STEPS_PER_EPOCH = np.ceil(1457/BATCH_SIZE)

# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.

def show_dataset(labeled_ds):
    for image, label in labeled_ds.take(1):
        print("Image shape: ", image.numpy().shape)
        print("Label: ", label.numpy())

def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
    # This is a small dataset, only load it once, and keep it in memory.
    # use `.cache(filename)` to cache preprocessing work for datasets that don't
    # fit in memory.
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

    ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    # Repeat forever
    ds = ds.repeat()
    ds = ds.map(augment, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)

    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds        

def show_batch(image_batch, label_batch):
    plt.figure(figsize=(10,10))
    for n in range(25):
        ax = plt.subplot(5,5,n+1)
        plt.imshow(image_batch[n])
        plt.title(CLASS_NAMES[label_batch[n]==1][0].title())
        plt.axis('off')

def convert(image, label):
  image = tf.image.convert_image_dtype(image, tf.float32) # Cast and normalize the image to [0,1]
  return image, label

def augment(image,label):
  image,label = convert(image, label)
  image = tf.image.convert_image_dtype(image, tf.float32)
  image = tf.image.random_brightness(image, max_delta=0.3)
  image = tf.image.random_contrast(image, 0.8, 1.2)
  #image = tf.image.random_crop(image, size=[224, 224, 3]) # Random crop 
  image = tf.image.random_flip_left_right(image) 
  image = tf.image.random_flip_up_down(image, seed=None)
  image = tf.image.random_hue(image, max_delta= 0.05, seed=None)
  image = tf.image.random_saturation(image, 0.7, 1.3, seed=None)

  return image,label
def model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(4))
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.summary()
    return model

def train(train_data, val_data, epoch_num, steps_per_epoch):
    history = model.fit(x = train_data,
                            steps_per_epoch = steps_per_epoch,
                            epochs = epoch_num,
                            validation_data = val_data,
                            validation_steps = 364//BATCH_SIZE)
    return history

def show_loss_acc(history, epochs):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    
    loss=history.history['loss']
    val_loss=history.history['val_loss']
    
    epochs_range = range(epochs)
    
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()
       
if(__name__ == "__main__"):
    train_list_ds, val_list_ds = load_data(data_dir)
    #show_img(train_images)
    #show_data(train_list_ds)
    train_labeled_ds = train_list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    val_labeled_ds = val_list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    show_dataset(train_labeled_ds)
    augmented_train_ds = prepare_for_training(train_labeled_ds)
    val_ds = prepare_for_training(val_labeled_ds)
    #train_image_batch, train_label_batch = next(iter(augmented_train_ds))
    #show_batch(train_image_batch.numpy(), train_label_batch.numpy())
    model = model()
    epoch_num = 30
    history = train(augmented_train_ds, val_ds, epoch_num=epoch_num, steps_per_epoch=STEPS_PER_EPOCH)
    show_loss_acc(history, epoch_num)







