from tensorflow import keras
from keras import layers


class InvertedResidualBlock(layers.Layer):
  def __init__(self, expand_ratio, out_channels, stride, **kwargs):
    super(InvertedResidualBlock, self).__init__(**kwargs)
    self.expand_ratio = expand_ratio
    self.out_channels = out_channels
    self.stride = stride

    self.use_shortcut = stride == 1 and out_channels == out_channels

    self.conv1 = layers.Conv2D(filters=in_channels * expand_ratio, kernel_size=1, use_bias=False)
    self.bn1 = layers.BatchNormalization()
    self.relu1 = layers.ReLU()

    self.depthwise_conv = layers.DepthwiseConv2D(kernel_size=3, strides=stride, padding="same", use_bias=False)
    self.bn2 = layers.BatchNormalization()
    self.relu2 = layers.ReLU()

    self.pointwise_conv = layers.Conv2D(filters=out_channels, kernel_size=1, use_bias=False)
    self.bn3 = layers.BatchNormalization()

  def call(self, inputs, training=False):
    x = inputs
    if self.use_shortcut:
      shortcut = inputs
    else:
      shortcut = layers.Conv2D(filters=out_channels, kernel_size=1, strides=stride, use_bias=False)(inputs)
      shortcut = layers.BatchNormalization()(shortcut)

    x = self.conv1(x)
    x = self.bn1(x, training=training)
    x = self.relu1(x)

    x = self.depthwise_conv(x)
    x = self.bn2(x, training=training)
    x = self.relu2(x)

    x = self.pointwise_conv(x)
    x = self.bn3(x, training=training)

    x = shortcut + x
    return x


def MobileNetV2(input_shape=(224, 224, 3), num_classes=1000):
  inputs = layers.Input(shape=input_shape)

  x = layers.Conv2D(filters=32, kernel_size=3, strides=2, padding="same", use_bias=False)(inputs)
  x = layers.BatchNormalization()(x)
  x = layers.ReLU()(x)

  # Inverted residual blocks
  x = InvertedResidualBlock(expand_ratio=1, out_channels=16, stride=1)(x)
  x = InvertedResidualBlock(expand_ratio=6, out_channels=24, stride=2)(x)
  x = InvertedResidualBlock(expand_ratio=6, out_channels=32, stride=2)(x)
  x = InvertedResidualBlock(expand_ratio=6, out_channels=64, stride=2)(x)
  x = InvertedResidualBlock(expand_ratio=6, out_channels=96, stride=1)(x)
  x = InvertedResidualBlock(expand_ratio=6, out_channels=160, stride=2)(x)
  x = InvertedResidualBlock(expand_ratio=6, out_channels=320, stride=1)(x)

  # Global average pooling
  x = layers.GlobalAveragePooling2D()(x)

  # Output layer
  outputs = layers.Dense(num_classes, activation="softmax")(x)

  model = keras.Model(inputs=inputs, outputs=outputs)
  return model


from keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import Callback

class SegmentationCallback(Callback):
    def __init__(self, monitor='tran_loss', save_best_only=True,
                 checkpoint_path='best_model.h5', log_dir='./logs',
                 other=True, patience=4, **args):
        # we define the inisilizer of the parent class Callback
        super(SegmentationCallback, self).__init__(**args)
        self.monitor = monitor
        self.patience = patience
        self.checkpoint_path = checkpoint_path
        self.log_dir = log_dir

        # Define the EarlyStopping Object
        self.early_stopping = EarlyStopping(monitor=self.monitor, patience=self.patience)

        # Define the ModelCheckpoint Object
        self.model_checkpoint = ModelCheckpoint(filepath=self.checkpoint_path, monitor=self.monitor, save_best_only=True)

        # Define the TensorBoard Object
        self.tensorboard = TensorBoard(log_dir=self.log_dir)


    def on_train_begin(self, logs=None):
        self.best_val_loss = float('inf')


def on_epoch_end(self, epoch, logs=None):
    current_val_loss = logs.get(self.monitor)
    if current_val_loss is None:
        return

    self.early_stopping.on_epoch_end(epoch, logs)
    self.model_checkpoint.on_epoch_end(epoch, logs)
    self.tensorboard.on_epoch_end(epoch, logs)

    if current_val_loss < self.best_val_loss:
        self.best_val_loss = current_val_loss
    else:
        if self.early_stopping.stopped_epoch >= epoch - self.patience:
            self.model.stop_training = True


    def on_train_end(self, logs=None):
        self.tensorboard.on_train_end(logs)
