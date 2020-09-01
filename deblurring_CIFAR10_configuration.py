import tensorflow as tf
from utilities import get_metrics, get_other_metrics, get_model, get_loss, SSIMLoss, PSNR
tf.keras.backend.set_floatx('float64')

demo = False
model_name = 'SkipConnections_v2'
loss_name = 'SSIMLoss' # mse, mae, SSIMLoss, PSNR
EPOCHS = 50
other_EPOCHS = 15

test_lower_bound = 10 #parameters to limit the test prediction on a subset of the dataset
test_upper_bound = 20

metrics = get_metrics(loss_name)
other_metrics = get_other_metrics(metrics)
model = get_model(model_name)
loss = get_loss(loss_name)
