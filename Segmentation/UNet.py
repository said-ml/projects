
'''
author: said koussi
in the process 
'''
import numpy as np
import tensorflow as tf
from tensorflow import keras

from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import Conv2D
from keras.layers import UpSampling2D
from keras.layers import Dropout
from keras.layers import Layer

from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy
from keras.metrics import binary_accuracy
from keras.models import Model
from keras.utils import plot_model

import tensorflow as tf
from tensorflow.keras import layers

# Define the encoder block
class EncoderBlock(layers.Layer):
    def __init__(self, filters, kernel_size):
        super(EncoderBlock, self).__init__()
        self.conv = layers.Conv2D(filters, kernel_size, activation='relu', padding='same')
        self.pool = layers.MaxPooling2D(pool_size=(2, 2))

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.pool(x)
        return x

# Define the decoder block
class DecoderBlock(layers.Layer):
    def __init__(self, filters, kernel_size):
        super(DecoderBlock, self).__init__()
        self.up = layers.UpSampling2D(size=(2, 2))
        self.conv = layers.Conv2D(filters, kernel_size, activation='relu', padding='same')

    def call(self, inputs, skip):
        x = self.up(inputs)
        x = tf.concat([x, skip], axis=-1)
        x = self.conv(x)
        return x

# Define the UNet model
class UNet(tf.keras.Model):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.encoder1 = EncoderBlock(64, 3)
        self.encoder2 = EncoderBlock(128, 3)
        self.encoder3 = EncoderBlock(256, 3)
        self.encoder4 = EncoderBlock(512, 3)
        self.center = layers.Conv2D(1024, 3, activation='relu', padding='same')
        self.decoder4 = DecoderBlock(512, 3)
        self.decoder3 = DecoderBlock(256, 3)
        self.decoder2 = DecoderBlock(128, 3)
        self.decoder1 = DecoderBlock(64, 3)
        self.output_conv = layers.Conv2D(num_classes, 1, activation='softmax')

    def call(self, inputs):
        x1 = self.encoder1(inputs)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        center = self.center(x4)
        x = self.decoder4(center, x4)
        x = self.decoder3(x, x3)
        x = self.decoder2(x, x2)
        x = self.decoder1(x, x1)
        output = self.output_conv(x)
        return output

# Create an instance of the UNet model
model = UNet(num_classes=2)  # Set num_classes to the number of segmentation classes

# Compile the model (set appropriate loss and metrics for your task)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print the model summary
#json_model=model.to_json()
#print(json_model)

# Ploting the Architecture of the model
#plot_model(model, to_file='UNet_Architecture.png', show_shapes=True,
#          show_dtype=True, show_layer_names=True,
#           show_layer_activations=True)


###########################===============================########################

from tensorflow import keras
from tensorflow.keras import layers
def get_model(img_size, num_classes):
 inputs = keras.Input(shape=img_size + (3,))
 x = layers.Rescaling(1./255)(inputs)
 x = layers.Conv2D(64, 3, strides=2, activation="relu", padding="same")(x)
 x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
 x = layers.Conv2D(128, 3, strides=2, activation="relu", padding="same")(x)
 x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
 x = layers.Conv2D(256, 3, strides=2, padding="same", activation="relu")(x)
 x = layers.Conv2D(256, 3, activation="relu", padding="same")(x)
 x = layers.Conv2DTranspose(256, 3, activation="relu", padding="same")(x)
 x = layers.Conv2DTranspose(
 256, 3, activation="relu", padding="same", strides=2)(x)
 x = layers.Conv2DTranspose(128, 3, activation="relu", padding="same")(x)
 x = layers.Conv2DTranspose(
 128, 3, activation="relu", padding="same", strides=2)(x)
 x = layers.Conv2DTranspose(64, 3, activation="relu", padding="same")(x)
 x = layers.Conv2DTranspose(
 64, 3, activation="relu", padding="same", strides=2)(x)
 outputs = layers.Conv2D(num_classes, 3, activation="softmax", 
padding="same")(x)
 model = keras.Model(inputs, outputs)
 return model
model = get_model(img_size=img_size, num_classes=3)
model.summary()

          ################-------Define The UNet mode---------l################
'''
class UNet(Model):

    def __init__(self, num_classes):
        super(UNet, self).__init__()

        # Define The Encoder Block
        self.encoder1=Encoder(64,3)
        self.encoder2=Encoder(128, 3)
        self.encoder3=Encoder(256, 3)
        self.encoder4=Encoder(512, 3)

        # Define the center
        self.center=Conv2D(1024, 3, activatio='reu', padding='same')

        # Define the Decoder block
        self.decoder4=Decoder(512, 3)
        self.decoder3 = Decoder(256, 3)
        self.decoder2 = Decoder(128, 3)
        self.decoder1 = Decoder(64, 3)

    def call(self, inputs):

        e1=self.encoder1(inputs)
        e2=self.encoder2(e1)
        e3=self.encoder3(e2)
        e4=self.encoder4(e3)

        center=self.center(e4)

        x=self.decoder(center, e4)
        x=self.decoder(x, e3)
        x=self.decoder(x, e2)
        x=self.decoder(x, e1)

        output=self.outputs(x)
        return output

if __name__=='__init__':

    # create the instance object of the UNet model
    unet=UNet(2)
    # compile the model
    unet.compile(loss=BinaryCrossentropy, optimizer=Adam, metrics=binary_accuracy)
    # getting the summary of the UNet model
    unet.summary()
'''

