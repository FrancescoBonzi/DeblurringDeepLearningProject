
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from utilities import ResnetLayer

########################################
### DEFINITION OF THE NEURAL NETWORK ###
########################################


class DeblurringCNNBase_v1(tf.keras.Model):
    def __init__(self):
        super(DeblurringImagesModel, self).__init__()
        input_shape = (None, height, width, 3)
        self.conv1 = Conv2D(32, 3, activation='relu', input_shape=input_shape)
        self.conv2 = Conv2D(64, 3, activation='relu')
        self.conv3 = Conv2D(128, 3, activation='relu')
        self.deconv1 = Conv2DTranspose(128, 3, activation='relu')
        self.deconv2 = Conv2DTranspose(64, 3, activation='relu')
        self.deconv3 = Conv2DTranspose(32, 3, activation='relu')
        self.output_layer = Conv2DTranspose(3, 3, activation='relu', padding='same')

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.output_layer(x)
        return x

class DeblurringResnet(tf.keras.Model):
    def __init__(self):
        super(DeblurringResnet, self).__init__()
        self.conv1 = Conv2D(8, 5, strides = 2, activation='relu', input_shape=input_shape)
        self.resnet1 = ResnetLayer(num_filters=8, kernel_size=5)
        self.resnet2 = ResnetLayer(num_filters=8, kernel_size=5)
        self.conv2 = Conv2D(16, 5, strides = 2, activation='relu', input_shape=input_shape)
        self.resnet3 = ResnetLayer(num_filters=16, kernel_size=5)
        self.resnet4 = ResnetLayer(num_filters=16, kernel_size=5)
        self.conv3 = Conv2D(32, 3, strides = 2, activation='relu', input_shape=input_shape)
        self.resnet5 = ResnetLayer(num_filters=32, kernel_size=3)

        self.resnet6 = ResnetLayer(num_filters=32, kernel_size=3)
        self.deconv1 = Conv2DTranspose(32, 3, strides = 2, activation='relu', input_shape=input_shape)
        self.resnet7 = ResnetLayer(num_filters=32, kernel_size=5)
        self.resnet8 = ResnetLayer(num_filters=32, kernel_size=5)
        self.deconv2 = Conv2DTranspose(16, 5, strides = 2, activation='relu', input_shape=input_shape)
        self.resnet9 = ResnetLayer(num_filters=16, kernel_size=5)
        self.resnet10 = ResnetLayer(num_filters=16, kernel_size=5)
        self.deconv3 = Conv2DTranspose(8, 8, strides = 2, activation='relu', input_shape=input_shape)
        self.output_layer = Conv2DTranspose(3, 3, activation='relu', padding='same')

    def call(self, x):
        x = self.conv1(x)
        x = self.resnet1(x)
        x = self.resnet2(x)
        x = self.conv2(x)
        x = self.resnet3(x)
        x = self.resnet4(x)
        x = self.conv3(x)
        x = self.resnet5(x)
        x = self.resnet6(x)
        x = self.deconv1(x)
        x = self.resnet7(x)
        x = self.resnet8(x)
        x = self.deconv2(x)
        x = self.resnet9(x)
        x = self.resnet10(x)
        x = self.deconv3(x)
        output = self.output_layer(x)
        return output