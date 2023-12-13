
from tensorflow.keras.layers import Layer
import tensorflow as tf

class Augment(Layer):
    """
    Data augmentation layer for images.

    Args:
        horizontal_flip (bool): Whether to randomly flip images horizontally.
        vertical_flip (bool): Whether to randomly flip images vertically.
        rotation_range (float): Range for random rotation in degrees.
        zoom_range (tuple): Range for random zooming.
        shear_range (float): Range for random shearing in degrees.
        brightness_range (tuple): Range for random brightness adjustment.
        contrast_range (tuple): Range for random contrast adjustment.
    """

    def __init__(self,
                 horizontal_flip=False,
                 vertical_flip=False,
                 rotation_range=0.0,
                 zoom_range=(1.0, 1.0),
                 shear_range=0.0,
                 brightness_range=(1.0, 1.0),
                 contrast_range=(1.0, 1.0),
                 **kwargs):

        # first we define the inilizer of the parent class (Layer)
        super(Augment, self).__init__(**kwargs)
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rotation_range = rotation_range
        self.zoom_range = zoom_range
        self.shear_range = shear_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range

    def call(self, inputs, label, training=None):
        if training is not None and not training:
            return inputs

        image = inputs
        segmentation_mask=label

        # Random horizontal flip
        if self.horizontal_flip:
            image = tf.image.random_flip_left_right(image)
            segmentation_mask= tf.image.random_flip_left_right(segmentation_mask)

        # Random vertical flip
        if self.vertical_flip:
            image = tf.image.random_flip_up_down(image)
            segmentation_mask = tf.image.random_flip_up_down(segmentation_mask)

        # Random rotation
        if self.rotation_range > 0.0:
            angle = tf.random.uniform([], -self.rotation_range, self.rotation_range)
            image = tf.image.rot90(image, k=tf.cast(angle, tf.int32))
            segmentation_mask = tf.image.rot90(segmentation_mask, k=tf.cast(angle, tf.int32))

        # Random zoom
        if self.zoom_range[0] != 1.0 or self.zoom_range[1] != 1.0:
            zoom = tf.random.uniform([], self.zoom_range[0], self.zoom_range[1], dtype=tf.float32)
            image = tf.image.resize(image, tf.cast(tf.shape(image)[:2] * zoom, tf.int32))
            segmentation_mask = tf.image.resize(segmentation_mask, tf.cast(tf.shape(segmentation_mask)[:2] * zoom, tf.int32))

        # Random shear
        if self.shear_range > 0.0:
            shear = tf.random.uniform([], -self.shear_range, self.shear_range)
            image = tf.image.shear_x(image, shear)
            segmentation_mask = tf.image.shear_x(segmentation_mask , shear)

        # Random brightness
        if self.brightness_range[0] != 1.0 or self.brightness_range[1] != 1.0:
            brightness = tf.random.uniform([], self.brightness_range[0], self.brightness_range[1])
            image = tf.image.adjust_brightness(image, brightness)
            segmentation_mask = tf.image.adjust_brightness(segmentation_mask, brightness)

        # Random contrast
        if self.contrast_range[0] != 1.0 or self.contrast_range[1] != 1.0:
            contrast = tf.random.uniform([], self.contrast_range[0], self.contrast_range[1])
            image = tf.image.adjust_contrast(image, contrast)
            segmentation_mask=tf.image.adjust_contrast(segmentation_mask, contrast)

        return image, segmentation_mask

if __name__=='__main__':
    '''
    chaeck out you code here
    '''
