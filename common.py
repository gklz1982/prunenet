import os
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
import lmdb
import caffe.proto.caffe_pb2
from caffe.io import datum_to_array


DATA_PATH = '../data/imagenet'
MODEL_FILE = '../models/bvlc_reference_caffenet/deploy.prototxt'
WEIGHTS_FILE = '../models/bvlc_reference_caffenet/caffenet_train_iter_33100.caffemodel'
MEAN_FILE = '../data/imagenet/mean.lmdb'

blob = caffe.proto.caffe_pb2.BlobProto()
data = open( MEAN_FILE , 'rb' ).read()
blob.ParseFromString(data)
mean_arr = np.array( caffe.io.blobproto_to_array(blob) )[0]


sign = lambda x: np.sign(np.maximum(0., x)).astype(np.float32)

(th, tw) = (227, 227)

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
    return img


def load_lmdb(path):
    '''
    given the path to imdb data. return (images, labels) pair.
    all data are read in memory for fast processing.
    not feasible for large datasets.
    '''
    lmdb_env = lmdb.open(path)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    datum = caffe.proto.caffe_pb2.Datum()
    data_list = []
    data_label = []
    for key, value in lmdb_cursor:
        datum.ParseFromString(value)
        img = caffe.io.datum_to_array(datum).astype(np.float32)
        img = preprocess(img)
        data_label.append(datum.label)
        data_list.append(img)
    return (data_list, data_label)


if not os.environ.get('SKIP_TRAINING_DATA'):
    with Timer('loading training data'):
        (train_list, train_label) = load_lmdb(DATA_PATH + '/train10.lmdb')


if not os.environ.get('SKIP_TEST_DATA'):
    with Timer('loading test data'):
        (test_list, test_label) = load_lmdb(DATA_PATH + '/test10.lmdb')
        print test_list[0]


def sample_minibatch(train_list, batch_size):
    minibatch_list = choice(train_list, batch_size)
    im_input = np.zeros((batch_size, 3, th, tw))
    for (pi, img) in enumerate(minibatch_list):
        im_input[pi, :, :, :] = img
    return im_input

def sparsity(blob_data):
    return float(np.sum(np.abs(blob_data) > 0)) / np.prod(blob_data.shape)

