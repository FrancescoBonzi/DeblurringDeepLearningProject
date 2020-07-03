import tensorflow as tf
from utilities import get_metrics, get_other_metrics, get_model, get_loss, SSIM, PSNR
tf.keras.backend.set_floatx('float64')

demo = True
model_name = 'CNNBase_v1'
loss_name = 'SSIM' # mse, mae, SSIM, PSNR
EPOCHS = 1

metrics = get_metrics(loss_name)
other_metrics = get_other_metrics(metrics)
model = get_model(model_name)
loss = get_loss(loss_name)