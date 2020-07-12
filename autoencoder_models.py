
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization


##############################
### RESIDUAL NETWORK LAYER ###
##############################


class ResnetLayer(tf.keras.layers.Layer):
    def __init__(self,
                 num_filters=16,
                 kernel_size=3):
        super(ResnetLayer, self).__init__()
        self.conv1 = Conv2D(num_filters, kernel_size, activation='relu', padding='same')
        self.conv2 = Conv2D(num_filters, kernel_size, padding='same')

    def call(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        output = tf.keras.activations.relu(x + y)
        return output

        
########################################
### DEFINITION OF THE NEURAL NETWORK ###
########################################


class DeblurringCNNBase_v1(tf.keras.Model):
    def __init__(self):
        super(DeblurringCNNBase_v1, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.conv2 = Conv2D(64, 3, activation='relu')
        self.conv3 = Conv2D(128, 3, activation='relu')
        self.deconv1 = Conv2DTranspose(128, 3, activation='relu')
        self.deconv2 = Conv2DTranspose(64, 3, activation='relu')
        self.deconv3 = Conv2DTranspose(32, 3, activation='relu')
        self.output_layer = Conv2DTranspose(
            3, 3, activation='relu', padding='same')

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.output_layer(x)
        return x


class DeblurringCNNBase_v2(tf.keras.Model):
    def __init__(self):
        super(DeblurringCNNBase_v2, self).__init__()
        self.conv1 = Conv2D(8, 3, activation='relu')
        self.conv2 = Conv2D(8, 3, activation='relu', padding='same')
        self.conv3 = Conv2D(8, 3, activation='relu', padding='same')
        self.conv4 = Conv2D(8, 3, activation='relu', padding='same')
        self.conv5 = Conv2D(8, 3, activation='relu', padding='same')
        self.conv6 = Conv2D(16, 3, activation='relu')
        self.conv7 = Conv2D(16, 3, activation='relu', padding='same')
        self.conv8 = Conv2D(16, 3, activation='relu', padding='same')
        self.conv9 = Conv2D(16, 3, activation='relu', padding='same')
        self.conv10 = Conv2D(16, 3, activation='relu', padding='same')
        self.conv11 = Conv2D(32, 3, activation='relu')
        self.conv12 = Conv2D(32, 3, activation='relu', padding='same')
        self.conv13 = Conv2D(32, 3, activation='relu', padding='same')
        self.conv14 = Conv2D(32, 3, activation='relu', padding='same')
        self.conv15 = Conv2D(32, 3, activation='relu', padding='same')

        self.deconv1 = Conv2DTranspose(32, 3, activation='relu')
        self.deconv2 = Conv2DTranspose(32, 3, activation='relu', padding='same')
        self.deconv3 = Conv2DTranspose(32, 3, activation='relu', padding='same')
        self.deconv4 = Conv2DTranspose(32, 3, activation='relu', padding='same')
        self.deconv5 = Conv2DTranspose(32, 3, activation='relu', padding='same')
        self.deconv6 = Conv2DTranspose(16, 3, activation='relu')
        self.deconv7 = Conv2DTranspose(16, 3, activation='relu', padding='same')
        self.deconv8 = Conv2DTranspose(16, 3, activation='relu', padding='same')
        self.deconv9 = Conv2DTranspose(16, 3, activation='relu', padding='same')
        self.deconv10 = Conv2DTranspose(16, 3, activation='relu', padding='same')
        self.deconv11 = Conv2DTranspose(8, 3, activation='relu')
        self.deconv12 = Conv2DTranspose(8, 3, activation='relu', padding='same')
        self.deconv13 = Conv2DTranspose(8, 3, activation='relu', padding='same')
        self.deconv14 = Conv2DTranspose(8, 3, activation='relu', padding='same')
        self.deconv15 = Conv2DTranspose(8, 3, activation='relu', padding='same')
        self.output_layer = Conv2DTranspose(
            3, 3, activation='relu', padding='same')

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.conv14(x)
        x = self.conv15(x)

        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)
        x = self.deconv6(x)
        x = self.deconv7(x)
        x = self.deconv8(x)
        x = self.deconv9(x)
        x = self.deconv10(x)
        x = self.deconv11(x)
        x = self.deconv12(x)
        x = self.deconv13(x)
        x = self.deconv14(x)
        x = self.deconv15(x)
        
        x = self.output_layer(x)
        return x


class DeblurringSkipConnections(tf.keras.Model):
    def __init__(self):
        super(DeblurringSkipConnections, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.conv2 = Conv2D(64, 3, activation='relu')
        self.conv3 = Conv2D(64, 3, activation='relu')
        self.conv4 = Conv2D(128, 3, activation='relu')
        self.deconv1 = Conv2DTranspose(128, 3, activation='relu')
        self.deconv2 = Conv2DTranspose(64, 3, activation='relu')
        self.deconv3 = Conv2DTranspose(32, 3, activation='relu')
        self.deconv4 = Conv2DTranspose(3, 3, activation='relu')

    def call(self, input_img):
        c1 = self.conv1(input_img)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        d1 = self.deconv1(c4)
        d2 = self.deconv2(d1)
        d3 = self.deconv3(d2 + c2)
        d4 = self.deconv4(d3)
        output_img = d4 + input_img
        return output_img
        

class DeblurringResnet_v1(tf.keras.Model):
    def __init__(self):
        super(DeblurringResnet_v1, self).__init__()
        self.conv1 = Conv2D(32, 5, strides=2, activation='relu')
        self.resnet1 = ResnetLayer(num_filters=32, kernel_size=5)
        self.conv2 = Conv2D(64, 5, strides=2, activation='relu')
        self.resnet2 = ResnetLayer(num_filters=64, kernel_size=5)
        self.conv3 = Conv2D(128, 3, strides=2, activation='relu')
        self.resnet3 = ResnetLayer(num_filters=128, kernel_size=3)

        self.deconv1 = Conv2DTranspose(128, 3, strides=2, activation='relu')
        self.resnet4 = ResnetLayer(num_filters=128, kernel_size=5)
        self.deconv2 = Conv2DTranspose(64, 5, strides=2, activation='relu')
        self.resnet5 = ResnetLayer(num_filters=64, kernel_size=5)
        self.deconv3 = Conv2DTranspose(32, 8, strides=2, activation='relu')
        self.output_layer = Conv2DTranspose(
            3, 3, activation='relu', padding='same')

    def call(self, x):
        x = self.conv1(x)
        x = self.resnet1(x)
        x = self.conv2(x)
        x = self.resnet2(x)
        x = self.conv3(x)
        x = self.resnet3(x)
        x = self.deconv1(x)
        x = self.resnet4(x)
        x = self.deconv2(x)
        x = self.resnet5(x)
        x = self.deconv3(x)
        output = self.output_layer(x)
        return output


class DeblurringResnet_v2(tf.keras.Model):
    def __init__(self):
        super(DeblurringResnet_v2, self).__init__()
        self.conv1 = Conv2D(8, 5, strides=2, activation='relu')
        self.resnet1 = ResnetLayer(num_filters=8, kernel_size=5)
        self.resnet2 = ResnetLayer(num_filters=8, kernel_size=5)
        self.conv2 = Conv2D(16, 5, strides=2, activation='relu')
        self.resnet3 = ResnetLayer(num_filters=16, kernel_size=5)
        self.resnet4 = ResnetLayer(num_filters=16, kernel_size=5)
        self.conv3 = Conv2D(32, 3, strides=2, activation='relu')
        self.resnet5 = ResnetLayer(num_filters=32, kernel_size=3)

        self.resnet6 = ResnetLayer(num_filters=32, kernel_size=3)
        self.deconv1 = Conv2DTranspose(32, 3, strides=2, activation='relu')
        self.resnet7 = ResnetLayer(num_filters=32, kernel_size=5)
        self.resnet8 = ResnetLayer(num_filters=32, kernel_size=5)
        self.deconv2 = Conv2DTranspose(16, 5, strides=2, activation='relu')
        self.resnet9 = ResnetLayer(num_filters=16, kernel_size=5)
        self.resnet10 = ResnetLayer(num_filters=16, kernel_size=5)
        self.deconv3 = Conv2DTranspose(8, 8, strides=2, activation='relu')
        self.output_layer = Conv2DTranspose(
            3, 3, activation='relu', padding='same')

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
