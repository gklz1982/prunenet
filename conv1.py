# experiment script.
# prune conv1.
import matplotlib
matplotlib.use('Agg')
import caffe
from PIL import Image

MODEL_FILE = 'models/bvlc_reference_caffenet/train_val.prototxt'
WEIGHTS_FILE = 'models/bvlc_reference_caffenet/caffenet_train_iter_19200.caffemodel'

net = caffe.Net(MODEL_FILE, WEIGHTS_FILE, caffe.TEST)

train/n03085013/n03085013_2061.JPEG



