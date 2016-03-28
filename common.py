import matplotlib
import matplotlib.pyplot as plt
import dill as pickle
import caffe
from caffe.io import oversample, resize_image
from scipy.misc import imresize
from PIL import Image
from utils import *
from theano.tensor.nnet import conv2d
import theano
import theano.tensor as T
from theano.tensor.signal.downsample import max_pool_2d
from optimizers import Adam


DATA_PATH = '../data/imagenet'
MODEL_FILE = '../models/bvlc_reference_caffenet/deploy.prototxt'
WEIGHTS_FILE = '../models/bvlc_reference_caffenet/caffenet_train_iter_33100.caffemodel'
MEAN_FILE = '../data/imagenet/mean.lmdb'

blob = caffe.proto.caffe_pb2.BlobProto()
data = open( MEAN_FILE , 'rb' ).read()
blob.ParseFromString(data)
mean_arr = np.array( caffe.io.blobproto_to_array(blob) )[0]


sign = lambda x: np.sign(np.maximum(0., x)).astype(np.float32)

(ih, iw) = (256, 256)
(th, tw) = (227, 227)

train_list = []
train_label = []
with open(DATA_PATH + '/train_list.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.replace('\n', '').split(' ')
        train_list.append(line[0])
        train_label.append(int(line[1]))


test_list = []
test_label = []
with open(DATA_PATH + '/test_list.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.replace('\n', '').split(' ')
        test_list.append(line[0])
        test_label.append(int(line[1]))


def preprocess(img):
    img = np.array(img, dtype=np.float32) # convert to single.
    if len(img.shape) == 2:
        old_img = img
        img = np.zeros(img.shape + (3,))
        img[:, :, 0] = old_img
        img[:, :, 1] = old_img
        img[:, :, 2] = old_img
    # first resize then crop.
    # TODO: it seems that using `resize_image` leads to lower accuracy: 0.53.
    # img = resize_image(img, (ih, iw))
    # img = np.array([img[:, :, 0], img[:, :, 1], img[:, :, 2]])
    img = img - mean_arr # substract mean.
# central crop.
    (h, w) = img.shape[1:]
    (ch, cw) = (int(h / 2), int(w / 2))
    (cth, ctw) = (int(th / 2), int(tw / 2))
    img = img[:, ch-cth:ch-cth+th, cw-ctw:cw-ctw+tw]
    #img = np.transpose(img, (1, 2, 0))

    ## only crop.
    #img = np.array([img[:, :, 0], img[:, :, 1], img[:, :, 2]])
    #(h, w) = img.shape[1:]
    #(ch, cw) = (int(h / 2), int(w / 2))
    #(cth, ctw) = (int(ih / 2), int(iw / 2))
    #img = img[:, ch-cth:ch-cth+ih, cw-ctw:cw-ctw+iw]
    #img = img - mean_arr # substract mean.
    #img = np.transpose(img, (1, 2, 0))
    #img = resize_image(img, (th, tw))
    #img = np.transpose(img, (2, 0, 1))
    return img


def sample_minibatch(train_list, batch_size):
    minibatch_list = choice(train_list, batch_size)
    im_input = np.zeros((batch_size, 3, th, tw))
    for (pi, img_path) in enumerate(minibatch_list):
        img = Image.open(DATA_PATH + '/' + img_path)
        img = preprocess(img)
        im_input[pi, :, :, :] = img
    return im_input

def sparsity(blob_data):
    return float(np.sum(np.abs(blob_data) > 0)) / np.prod(blob_data.shape)

