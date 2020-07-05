import tensorflow as tf
from utilities import get_metrics, get_other_metrics, get_model, get_loss, SSIMLoss, PSNR
tf.keras.backend.set_floatx('float64')

demo = True
model_name = 'SkipConnections'
loss_name = 'SSIMLoss' # mse, mae, SSIMLoss, PSNR
EPOCHS = 35

metrics = get_metrics(loss_name)
other_metrics = get_other_metrics(metrics)
model = get_model(model_name)
loss = get_loss(loss_name)