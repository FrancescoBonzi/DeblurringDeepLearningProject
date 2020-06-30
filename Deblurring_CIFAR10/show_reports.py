import dill
from utilities import inspect_report

EPOCHS = 35
filename = "./reports/CNN/" + "epochs" + str(EPOCHS) + ".obj"
filehandler = open(filename, 'rb') 
report = dill.load(filehandler)

inspect_report(report, ['loss', 'mae', 'mse', 'PSNR'])
