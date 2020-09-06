import tensorflow as tf
from scipy.ndimage import gaussian_filter
import time
import numpy as np
import PIL.Image
import IPython.display as display
import matplotlib.pyplot as plt

MAX_DIM = 256
NUM_STYLE_IMAGES = 50 

# Choose intermediate layers from the network to represent the style and content of the image:
content_layers = ['block5_conv2']

style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

style_weight = 1e-2
content_weight = 1e4

# Build a model that returns the style and content tensors.
class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        "Expects float input in [0,1]"
        inputs = inputs*255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(
            inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])

        style_outputs = [gram_matrix(style_output)
                         for style_output in style_outputs]

        content_dict = {content_name: value
                        for content_name, value
                        in zip(self.content_layers, content_outputs)}

        style_dict = {style_name: value
                      for style_name, value
                      in zip(self.style_layers, style_outputs)}

        return {'content': content_dict, 'style': style_dict}

# Visualize the input
def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

# Define a function to load an image and limit its maximum dimension to 256 pixels.
def load_img(path_to_img):
    max_dim = MAX_DIM
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    print(tf.shape(img))

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    print(shape)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

# Create a simple function to display an image:
def imshow(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)

    plt.imshow(image)
    if title:
        plt.title(title)

def cifar10_as_style_images():
    (style_images, _), (_, _) = tf.keras.datasets.cifar10.load_data()
    # Normalize pixel values to be between 0 and 1
    style_images = style_images / 255.0
    print(style_images.shape)

    # It takes the first image from cifar and uses it as content
    original_image = style_images[0]
    display.clear_output(wait=True)
    display.display(tensor_to_image(original_image))

    blur = 1.0
    tfimg = tf.reshape(style_images[0], [1, 32, 32, 3])
    orimg = np.array(tfimg)

    # Blurring
    r = gaussian_filter(orimg[:, :, :, 0], blur)
    r = r[:, :, :, np.newaxis]
    g = gaussian_filter(orimg[:, :, :, 1], blur)
    g = g[:, :, :, np.newaxis]
    b = gaussian_filter(orimg[:, :, :, 2], blur)
    b = b[:, :, :, np.newaxis]
    image_blurred = np.concatenate((r, g, b), axis=3)

    content_image = tf.reshape(image_blurred, [1, 32, 32, 3])
    content_image = tf.cast(content_image, tf.float32)

    style_images = tf.reshape(style_images[1:NUM_STYLE_IMAGES, :, :, :], [
                              NUM_STYLE_IMAGES-1, 1, 32, 32, 3])

    style_images = tf.cast(style_images, tf.float32)
    style_image = style_images[0]

    return content_image, style_image, style_images

def vgg_layers(layer_names):
    """ Creates a vgg model that returns a list of intermediate output values."""
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model

# Calculate style
# The content of an image is represented by the values of the intermediate feature maps.
# It turns out, the style of an image can be described by the means and correlations across the different feature maps.
# Calculate a Gram matrix that includes this information by taking the outer product of the feature vector with itself at each location,
# and averaging that outer product over all locations. This Gram matrix can be calculated for a particular layer as:

def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result/(num_locations)

# Since this is a float image, define a function to keep the pixel values between 0 and 1:
def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

def style_num_fin_content_loss(outputs, content_targets, style_targets, loss=-1):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2)
                           for name in style_outputs.keys()])
    style_loss *= style_weight / num_style_layers

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2)
                             for name in content_outputs.keys()])
    content_loss *= content_weight / num_content_layers
    if (loss == -1):
        loss = style_loss + content_loss
    else:
        loss = loss + style_loss + content_loss
    return loss

# Total variation loss
# One downside to this basic implementation is that it produces a lot of high frequency artifacts. Decrease these using an explicit regularization term on the high frequency components of the image. In style transfer, this is often called the total variation loss:
def high_pass_x_y(image):

    x_var = image[:, :, 1:, :] - image[:, :, :-1, :]
    y_var = image[:, 1:, :, :] - image[:, :-1, :, :]

    return x_var, y_var

@tf.function()
def train_step(image):
    content_targets = extractor(content_image)['content']

    loss = tf.cast(-1, tf.float32)
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        for styl_img in style_images:
            style_targets = extractor(styl_img)['style']

            loss = style_num_fin_content_loss(
                outputs, content_targets, style_targets, loss)

            loss = tf.math.add(loss, tf.reshape(total_variation_weight * tf.image.total_variation(image), ()))

        grad = tape.gradient(loss, image)
        opt.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))
    
content_image, style_image, style_images = cifar10_as_style_images()

'''plt.subplot(1, 2, 1)
imshow(content_image, 'Content Image')
plt.show()
display.clear_output(wait=True)
display.display(tensor_to_image(content_image))'''

'''plt.subplot(1, 2, 2)
imshow(style_image, 'Style Image')
display.clear_output(wait=True)
display.display(tensor_to_image(style_image))'''

# Now load a VGG19 without the classification head, and list the layer names
vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')

# And to create the model:
style_extractor_model = vgg_layers(style_layers)
style_outputs = style_extractor_model(style_image*255)

# When called on an image, this model returns the gram matrix (style) of the style_layers and content of the content_layers:
extractor = StyleContentModel(style_layers, content_layers)
results = extractor(tf.constant(content_image))

### Run gradient descent ###

# Create an optimizer. The paper recommends LBFGS, but Adam works okay, too:
opt = tf.keras.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

# Also, this high frequency component is basically an edge-detector. You can get similar output from the Sobel edge detector, for example:
sobel = tf.image.sobel_edges(content_image)
plt.subplot(1, 2, 1)
#imshow(clip_0_1(sobel[..., 0]/4+0.5), "Horizontal Sobel-edges")
plt.subplot(1, 2, 2)
#imshow(clip_0_1(sobel[..., 1]/4+0.5), "Vertical Sobel-edges")

# Choose a weight for the total_variation_loss:
total_variation_weight = 30

# Define a tf.Variable to contain the image to optimize. To make this quick, initialize it with the content image (the tf.Variable must be the same shape as the content image):
image = tf.Variable(content_image)
#tensor_to_image(image).save('./image' + str(0) + '.png')

# And run the optimization:
start = time.time()

epochs = 15
steps_per_epoch = 1  # 100

step = 0
for n in range(epochs):
    for m in range(steps_per_epoch):
        step += 1
        train_step(image)
        print(".", end='')
    tensor_to_image(image).save('./image' + str(int(step/steps_per_epoch)) + '.png')
    print("Train step: {}".format(step))

end = time.time()
print("Total time: {:.1f}".format(end-start))

